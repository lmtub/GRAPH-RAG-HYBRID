static int avisynth_read_header(AVFormatContext *s)

{

  AVISynthContext *avs = s->priv_data;

  HRESULT res;

  AVIFILEINFO info;

  DWORD id;

  AVStream *st;

  AVISynthStream *stream;

  wchar_t filename_wchar[1024] = { 0 };

  char filename_char[1024] = { 0 };



  AVIFileInit();



  /* avisynth can't accept UTF-8 filename */

  MultiByteToWideChar(CP_UTF8, 0, s->filename, -1, filename_wchar, 1024);

  WideCharToMultiByte(CP_THREAD_ACP, 0, filename_wchar, -1, filename_char, 1024, NULL, NULL);

  res = AVIFileOpen(&avs->file, filename_char, OF_READ|OF_SHARE_DENY_WRITE, NULL);

  if (res != S_OK)

    {

      av_log(s, AV_LOG_ERROR, "AVIFileOpen failed with error %ld", res);

      AVIFileExit();

      return -1;

    }



  res = AVIFileInfo(avs->file, &info, sizeof(info));

  if (res != S_OK)

    {

      av_log(s, AV_LOG_ERROR, "AVIFileInfo failed with error %ld", res);

      AVIFileExit();

      return -1;

    }



  avs->streams = av_mallocz(info.dwStreams * sizeof(AVISynthStream));



  for (id=0; id<info.dwStreams; id++)

    {

      stream = &avs->streams[id];

      stream->read = 0;

      if (AVIFileGetStream(avs->file, &stream->handle, 0, id) == S_OK)

        {

          if (AVIStreamInfo(stream->handle, &stream->info, sizeof(stream->info)) == S_OK)

            {

              if (stream->info.fccType == streamtypeAUDIO)

                {

                  WAVEFORMATEX wvfmt;

                  LONG struct_size = sizeof(WAVEFORMATEX);

                  if (AVIStreamReadFormat(stream->handle, 0, &wvfmt, &struct_size) != S_OK)

                    continue;



                  st = avformat_new_stream(s, NULL);

                  st->id = id;

                  st->codec->codec_type = AVMEDIA_TYPE_AUDIO;



                  st->codec->block_align = wvfmt.nBlockAlign;

                  st->codec->channels = wvfmt.nChannels;

                  st->codec->sample_rate = wvfmt.nSamplesPerSec;

                  st->codec->bit_rate = wvfmt.nAvgBytesPerSec * 8;

                  st->codec->bits_per_coded_sample = wvfmt.wBitsPerSample;



                  stream->chunck_samples = wvfmt.nSamplesPerSec * (uint64_t)info.dwScale / (uint64_t)info.dwRate;

                  stream->chunck_size = stream->chunck_samples * wvfmt.nChannels * wvfmt.wBitsPerSample / 8;



                  st->codec->codec_tag = wvfmt.wFormatTag;

                  st->codec->codec_id = ff_wav_codec_get_id(wvfmt.wFormatTag, st->codec->bits_per_coded_sample);

                }

              else if (stream->info.fccType == streamtypeVIDEO)

                {

                  BITMAPINFO imgfmt;

                  LONG struct_size = sizeof(BITMAPINFO);



                  stream->chunck_size = stream->info.dwSampleSize;

                  stream->chunck_samples = 1;



                  if (AVIStreamReadFormat(stream->handle, 0, &imgfmt, &struct_size) != S_OK)

                    continue;



                  st = avformat_new_stream(s, NULL);

                  st->id = id;

                  st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

                  st->r_frame_rate.num = stream->info.dwRate;

                  st->r_frame_rate.den = stream->info.dwScale;



                  st->codec->width = imgfmt.bmiHeader.biWidth;

                  st->codec->height = imgfmt.bmiHeader.biHeight;



                  st->codec->bits_per_coded_sample = imgfmt.bmiHeader.biBitCount;

                  st->codec->bit_rate = (uint64_t)stream->info.dwSampleSize * (uint64_t)stream->info.dwRate * 8 / (uint64_t)stream->info.dwScale;

                  st->codec->codec_tag = imgfmt.bmiHeader.biCompression;

                  st->codec->codec_id = ff_codec_get_id(ff_codec_bmp_tags, imgfmt.bmiHeader.biCompression);



                  st->duration = stream->info.dwLength;

                }

              else

                {

                  AVIStreamRelease(stream->handle);

                  continue;

                }



              avs->nb_streams++;



              st->codec->stream_codec_tag = stream->info.fccHandler;



              avpriv_set_pts_info(st, 64, info.dwScale, info.dwRate);

              st->start_time = stream->info.dwStart;

            }

        }

    }



  return 0;

}
