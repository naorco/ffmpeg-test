// ffmpeg_test.cpp : Defines the entry point for the console application.
//



extern "C"
{	
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include "stdafx.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// tutorial01.c
// Code based on a tutorial by Martin Bohme (boehme@inb.uni-luebeckREMOVETHIS.de)
// Tested on Gentoo, CVS version 5/01/07 compiled with GCC 4.1.1
// With updates from https://github.com/chelyaev/ffmpeg-tutorial
// Updates tested on:
// LAVC 54.59.100, LAVF 54.29.104, LSWS 2.1.101 
// on GCC 4.7.2 in Debian February 2015

// A small sample program that shows how to use libavformat and libavcodec to
// read video from a file.
//
// Use
//
// gcc -o tutorial01 tutorial01.c -lavformat -lavcodec -lswscale -lz
//
// to build (assuming libavformat and libavcodec are correctly installed
// your system).
//
// Run using
//
// tutorial01 myvideofile.mpg
//
// to write the first five frames from "myvideofile.mpg" to disk in PPM
// format.



void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) {
	FILE *pFile;
	char szFilename[32];
	int  y;

	// Open file
	sprintf_s(szFilename, "frame%d.ppm", iFrame);
	fopen_s(&pFile, szFilename, "wb");
	if (pFile == NULL)
		return;

	// Write header
	fprintf(pFile, "P6\n%d %d\n255\n", width, height);

	// Write pixel data
	for (y = 0; y<height; y++)
		fwrite(pFrame->data[0] + y*pFrame->linesize[0], 1, width * 3, pFile);

	// Close file
	fclose(pFile);
}

cv::Mat avframe_to_cvmat(AVFrame *frame)
{
	AVFrame frameBGR;
	cv::Mat m;

	memset(&frameBGR, 0, sizeof(frameBGR));

	int w = frame->width, h = frame->height;
	m = cv::Mat(h, w, CV_8UC3);
	//dst.data[0] = (uint8_t *)m.data;
	av_image_fill_arrays(frameBGR.data, frameBGR.linesize, m.data, AV_PIX_FMT_BGR24, w, h, 1);

	struct SwsContext *convert_ctx = NULL;
	AVPixelFormat src_pixfmt = (AVPixelFormat)frame->format;
	AVPixelFormat dst_pixfmt = AV_PIX_FMT_BGR24;
	convert_ctx = sws_getContext(w, h, src_pixfmt, w, h, dst_pixfmt,
		SWS_FAST_BILINEAR, NULL, NULL, NULL);
	sws_scale(convert_ctx, frame->data, frame->linesize, 0, h,
		frameBGR.data, frameBGR.linesize);
	sws_freeContext(convert_ctx);

	return m;
}
AVFrame cvmat_to_avframe(cv::Mat* frame)
{
	AVFrame dst;
	cv::Size frameSize = frame->size();
	AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
	AVFormatContext* outContainer = avformat_alloc_context();
	AVStream *outStream = avformat_new_stream(outContainer, encoder);
	avcodec_get_context_defaults3(outStream->codec, encoder);

	outStream->codec->pix_fmt = AV_PIX_FMT_YUV420P;
	outStream->codec->width = frame->cols;
	outStream->codec->height = frame->rows;
	av_image_fill_arrays(dst.data, dst.linesize, frame->data, AV_PIX_FMT_RGB24, outStream->codec->width, outStream->codec->height, 1);
	dst.width = frameSize.width;
	dst.height = frameSize.height;
	//SaveFrame(&dst, dst.width, dst.height, 0);
	return dst;
}
void FrameProcessAndDisplay(cv::Mat m)
{
	cuda::GpuMat d_src, d_dst_bi, d_dst_he, d_dst_bihe, d_tmp;
	vector<cuda::GpuMat> d_vec;
	Mat I = imread("baboon.jpg");
	if (I.empty())
		return;
	d_src.upload(I);
	cuda::bilateralFilter(d_src, d_dst_bi, -1, 50, 7);
	cuda::cvtColor(d_dst_bi, d_tmp, CV_BGR2YCrCb);
	cuda::split(d_tmp, d_vec);
	cuda::equalizeHist(d_vec[0], d_vec[0]);
	cuda::merge(d_vec, d_dst_bihe);
	cuda::cvtColor(d_dst_bihe, d_dst_bihe, CV_YCrCb2BGR);

	cuda::cvtColor(d_src, d_tmp, CV_BGR2YCrCb);
	cuda::split(d_tmp, d_vec);
	cuda::equalizeHist(d_vec[0], d_vec[0]);
	cuda::merge(d_vec, d_dst_he);
	cuda::cvtColor(d_dst_he, d_dst_he, CV_YCrCb2BGR);



	Mat dst_bi(d_dst_bi);
	Mat dst_he(d_dst_he);
	Mat dst_bihe(d_dst_bihe);
	namedWindow("bilateral", WINDOW_AUTOSIZE);
	namedWindow("hist eq", WINDOW_AUTOSIZE);
	namedWindow("hist eq after bilateral", WINDOW_AUTOSIZE);
	namedWindow("source", WINDOW_AUTOSIZE);
	imshow("bilateral", dst_bi);
	imshow("hist eq", dst_he);
	imshow("hist eq after bilateral", dst_bihe);

	imshow("source", I);
	if (cvWaitKey() == 27)
		return;
}


// compatibility with newer API
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif
int _tmain(int argc, const char * argv[])
{


	// Initalizing these to NULL prevents segfaults!
	AVFormatContext   *pFormatCtx = NULL;
	unsigned int i;
	int               videoStream;
	AVCodecContext    *pCodecCtxOrig = NULL;
	AVCodecContext    *pCodecCtx = NULL;
	AVInputFormat	  *fmt = NULL;
	AVCodec           *pCodec = NULL;
	AVFrame           *pFrame = NULL;
	AVFrame           *pFrameRGB = NULL;
	AVPacket          packet;
	int               frameFinished;
	int               numBytes;
	uint8_t           *buffer = NULL;
	struct SwsContext *sws_ctx = NULL;

	if (argc < 2) {
		printf("Please provide a movie file\n");
		return -1;
	}
	const char *letter = "udp://224.1.1.1:1234";
	char * format = "mpegts";
	fmt = av_find_input_format(format);
	
	// Register all formats and codecs
	av_register_all();


	if (avformat_network_init() < 0)
		return -1;

	// Open video file
	if (avformat_open_input(&pFormatCtx, letter, fmt, NULL) != 0)
		return -1; // Couldn't open file

	// Retrieve stream information
	if (avformat_find_stream_info(pFormatCtx, NULL)<0)
		return -1; // Couldn't find stream information

	// Dump information about file onto standard error
	av_dump_format(pFormatCtx, 0, letter, 0);

	// Find the first video stream
	videoStream = -1;
	for (i = 0; i<pFormatCtx->nb_streams; i++)
		if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
		videoStream = i;
		break;
		}
	if (videoStream == -1)
		return -1; // Didn't find a video stream

	// Get a pointer to the codec context for the video stream
	pCodecCtxOrig = pFormatCtx->streams[videoStream]->codec;
	// Find the decoder for the video stream
	pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
	if (pCodec == NULL) {
		fprintf(stderr, "Unsupported codec!\n");
		return -1; // Codec not found
	}
	// Copy context
	pCodecCtx = avcodec_alloc_context3(pCodec);
	if (avcodec_copy_context(pCodecCtx, pCodecCtxOrig) != 0) {
		fprintf(stderr, "Couldn't copy codec context");
		return -1; // Error copying codec context
	}

	// Open codec
	if (avcodec_open2(pCodecCtx, pCodec, NULL)<0)
		return -1; // Could not open codec

	// Allocate video frame
	pFrame = av_frame_alloc();

	// Allocate an AVFrame structure
	pFrameRGB = av_frame_alloc();
	if (pFrameRGB == NULL)
		return -1;
	
	// Determine required buffer size and allocate buffer
	numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecCtx->width,
		pCodecCtx->height,1);
	buffer = (uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

	// Assign appropriate parts of buffer to image planes in pFrameRGB
	// Note that pFrameRGB is an AVFrame, but AVFrame is a superset
	// of AVPicture
	
	av_image_fill_arrays(pFrameRGB->data,pFrameRGB->linesize, buffer, AV_PIX_FMT_RGB24,
		pCodecCtx->width, pCodecCtx->height,1);

	// initialize SWS context for software scaling
		sws_ctx = sws_getContext(pCodecCtx->width,
		pCodecCtx->height,
		pCodecCtx->pix_fmt,
		pCodecCtx->width,
		pCodecCtx->height,
		AV_PIX_FMT_RGB24,
		SWS_BILINEAR,
		NULL,
		NULL,
		NULL
		);

	// Read frames and save first five frames to disk
	i = 0;
	while (av_read_frame(pFormatCtx, &packet) >= 0) {
		// Is this a packet from the video stream?
		if (packet.stream_index == videoStream) {
			// Decode video frame
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

			// Did we get a video frame?
			if (frameFinished) {

				// convert to cv::MAT frame
				cv::Mat m = avframe_to_cvmat(pFrame);
				imshow("frame", m);
				cvWaitKey(30);
				// Convert the image from its native format to RGB
			/*	sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
					pFrame->linesize, 0, pCodecCtx->height,
					pFrameRGB->data, pFrameRGB->linesize);

				// Save the frame to disk
				if (++i <= 5)
					SaveFrame(pFrameRGB, pCodecCtx->width, pCodecCtx->height,
					i);*/
			}
		}

		// Free the packet that was allocated by av_read_frame
		av_packet_unref(&packet);
		
	}

	// Free the RGB image
	av_free(buffer);
	av_frame_free(&pFrameRGB);

	// Free the YUV frame
	av_frame_free(&pFrame);

	// Close the codecs
	avcodec_close(pCodecCtx);
	avcodec_close(pCodecCtxOrig);

	// Close the video file
	avformat_close_input(&pFormatCtx);

	return 0;
}







