static int hls_delete_old_segments(HLSContext *hls) {



    HLSSegment *segment, *previous_segment = NULL;

    float playlist_duration = 0.0f;

    int ret = 0, path_size, sub_path_size;

    char *dirname = NULL, *p, *sub_path;

    char *path = NULL;



    segment = hls->segments;

    while (segment) {

        playlist_duration += segment->duration;

        segment = segment->next;

    }



    segment = hls->old_segments;

    while (segment) {

        playlist_duration -= segment->duration;

        previous_segment = segment;

        segment = previous_segment->next;

        if (playlist_duration <= -previous_segment->duration) {

            previous_segment->next = NULL;

            break;

        }

    }



    if (segment && !hls->use_localtime_mkdir) {

        if (hls->segment_filename) {

            dirname = av_strdup(hls->segment_filename);

        } else {

            dirname = av_strdup(hls->avf->filename);

        }

        if (!dirname) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        p = (char *)av_basename(dirname);

        *p = '\0';

    }



    while (segment) {

        av_log(hls, AV_LOG_DEBUG, "deleting old segment %s\n",

                                  segment->filename);

        path_size =  (hls->use_localtime_mkdir ? 0 : strlen(dirname)) + strlen(segment->filename) + 1;

        path = av_malloc(path_size);

        if (!path) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        if (hls->use_localtime_mkdir)

            av_strlcpy(path, segment->filename, path_size);

        else { // segment->filename contains basename only

            av_strlcpy(path, dirname, path_size);

            av_strlcat(path, segment->filename, path_size);

        }



        if (unlink(path) < 0) {

            av_log(hls, AV_LOG_ERROR, "failed to delete old segment %s: %s\n",

                                     path, strerror(errno));

        }



        if (segment->sub_filename[0] != '\0') {

            sub_path_size = strlen(dirname) + strlen(segment->sub_filename) + 1;

            sub_path = av_malloc(sub_path_size);

            if (!sub_path) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }



            av_strlcpy(sub_path, dirname, sub_path_size);

            av_strlcat(sub_path, segment->sub_filename, sub_path_size);

            if (unlink(sub_path) < 0) {

                av_log(hls, AV_LOG_ERROR, "failed to delete old segment %s: %s\n",

                                         sub_path, strerror(errno));

            }

            av_free(sub_path);

        }

        av_freep(&path);

        previous_segment = segment;

        segment = previous_segment->next;

        av_free(previous_segment);

    }



fail:

    av_free(path);

    av_free(dirname);



    return ret;

}
