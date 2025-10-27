static int mxf_parse_structural_metadata(MXFContext *mxf)

{

    MXFPackage *material_package = NULL;

    MXFPackage *source_package = NULL;

    MXFPackage *temp_package = NULL;

    int i, j, k;



    dprintf("metadata sets count %d\n", mxf->metadata_sets_count);

    /* TODO: handle multiple material packages (OP3x) */

    for (i = 0; i < mxf->packages_count; i++) {

        if (!(temp_package = mxf_resolve_strong_ref(mxf, &mxf->packages_refs[i]))) {

            av_log(mxf->fc, AV_LOG_ERROR, "could not resolve package strong ref\n");

            return -1;

        }

        if (temp_package->type == MaterialPackage) {

            material_package = temp_package;

            break;

        }

    }

    if (!material_package) {

        av_log(mxf->fc, AV_LOG_ERROR, "no material package found\n");

        return -1;

    }



    for (i = 0; i < material_package->tracks_count; i++) {

        MXFTrack *material_track = NULL;

        MXFTrack *source_track = NULL;

        MXFTrack *temp_track = NULL;

        MXFDescriptor *descriptor = NULL;

        MXFStructuralComponent *component = NULL;

        const MXFCodecUL *codec_ul = NULL;

        const MXFCodecUL *container_ul = NULL;

        AVStream *st;



        if (!(material_track = mxf_resolve_strong_ref(mxf, &material_package->tracks_refs[i]))) {

            av_log(mxf->fc, AV_LOG_ERROR, "could not resolve material track strong ref\n");

            continue;

        }



        if (!(material_track->sequence = mxf_resolve_strong_ref(mxf, &material_track->sequence_ref))) {

            av_log(mxf->fc, AV_LOG_ERROR, "could not resolve material track sequence strong ref\n");

            return -1;

        }



        /* TODO: handle multiple source clips */

        for (j = 0; j < material_track->sequence->structural_components_count; j++) {

            /* TODO: handle timecode component */

            component = mxf_resolve_strong_ref(mxf, &material_track->sequence->structural_components_refs[j]);

            if (!component || component->type != SourceClip)

                continue;



            for (k = 0; k < mxf->packages_count; k++) {

                if (!(temp_package = mxf_resolve_strong_ref(mxf, &mxf->packages_refs[k]))) {

                    av_log(mxf->fc, AV_LOG_ERROR, "could not resolve source track strong ref\n");

                    return -1;

                }

                if (!memcmp(temp_package->package_uid, component->source_package_uid, 16)) {

                    source_package = temp_package;

                    break;

                }

            }

            if (!source_package) {

                av_log(mxf->fc, AV_LOG_ERROR, "material track %d: no corresponding source package found\n", material_track->track_id);

                break;

            }

            for (k = 0; k < source_package->tracks_count; k++) {

                if (!(temp_track = mxf_resolve_strong_ref(mxf, &source_package->tracks_refs[k]))) {

                    av_log(mxf->fc, AV_LOG_ERROR, "could not resolve source track strong ref\n");

                    return -1;

                }

                if (temp_track->track_id == component->source_track_id) {

                    source_track = temp_track;

                    break;

                }

            }

            if (!source_track) {

                av_log(mxf->fc, AV_LOG_ERROR, "material track %d: no corresponding source track found\n", material_track->track_id);

                break;

            }

        }

        if (!source_track)

            continue;



        st = av_new_stream(mxf->fc, source_track->track_id);

        st->priv_data = source_track;

        st->duration = component->duration;

        if (st->duration == -1)

            st->duration = AV_NOPTS_VALUE;

        st->start_time = component->start_position;

        av_set_pts_info(st, 64, material_track->edit_rate.num, material_track->edit_rate.den);



        if (!(source_track->sequence = mxf_resolve_strong_ref(mxf, &source_track->sequence_ref))) {

            av_log(mxf->fc, AV_LOG_ERROR, "could not resolve source track sequence strong ref\n");

            return -1;

        }



#ifdef DEBUG

        PRINT_KEY("data definition   ul", source_track->sequence->data_definition_ul);

#endif

        st->codec->codec_type = mxf_get_codec_type(mxf_data_definition_uls, &source_track->sequence->data_definition_ul);



        source_package->descriptor = mxf_resolve_strong_ref(mxf, &source_package->descriptor_ref);

        if (source_package->descriptor) {

            if (source_package->descriptor->type == MultipleDescriptor) {

                for (j = 0; j < source_package->descriptor->sub_descriptors_count; j++) {

                    MXFDescriptor *sub_descriptor = mxf_resolve_strong_ref(mxf, &source_package->descriptor->sub_descriptors_refs[j]);



                    if (!sub_descriptor) {

                        av_log(mxf->fc, AV_LOG_ERROR, "could not resolve sub descriptor strong ref\n");

                        continue;

                    }

                    if (sub_descriptor->linked_track_id == source_track->track_id) {

                        descriptor = sub_descriptor;

                        break;

                    }

                }

            } else

                descriptor = source_package->descriptor;

        }

        if (!descriptor) {

            av_log(mxf->fc, AV_LOG_INFO, "source track %d: stream %d, no descriptor found\n", source_track->track_id, st->index);

            continue;

        }

#ifdef DEBUG

        PRINT_KEY("essence codec     ul", descriptor->essence_codec_ul);

        PRINT_KEY("essence container ul", descriptor->essence_container_ul);

#endif

        /* TODO: drop PictureEssenceCoding and SoundEssenceCompression, only check EssenceContainer */

        codec_ul = mxf_get_codec_ul(mxf_codec_uls, &descriptor->essence_codec_ul);

        st->codec->codec_id = codec_ul->id;

        if (descriptor->extradata) {

            st->codec->extradata = descriptor->extradata;

            st->codec->extradata_size = descriptor->extradata_size;

        }

        if (st->codec->codec_type == CODEC_TYPE_VIDEO) {

            container_ul = mxf_get_codec_ul(mxf_picture_essence_container_uls, &descriptor->essence_container_ul);

            if (st->codec->codec_id == CODEC_ID_NONE)

                st->codec->codec_id = container_ul->id;

            st->codec->width = descriptor->width;

            st->codec->height = descriptor->height;

            st->codec->bits_per_sample = descriptor->bits_per_sample; /* Uncompressed */

            st->need_parsing = 2; /* only parse headers */

        } else if (st->codec->codec_type == CODEC_TYPE_AUDIO) {

            container_ul = mxf_get_codec_ul(mxf_sound_essence_container_uls, &descriptor->essence_container_ul);

            if (st->codec->codec_id == CODEC_ID_NONE)

                st->codec->codec_id = container_ul->id;

            st->codec->channels = descriptor->channels;

            st->codec->bits_per_sample = descriptor->bits_per_sample;

            st->codec->sample_rate = descriptor->sample_rate.num / descriptor->sample_rate.den;

            /* TODO: implement CODEC_ID_RAWAUDIO */

            if (st->codec->codec_id == CODEC_ID_PCM_S16LE) {

                if (descriptor->bits_per_sample == 24)

                    st->codec->codec_id = CODEC_ID_PCM_S24LE;

                else if (descriptor->bits_per_sample == 32)

                    st->codec->codec_id = CODEC_ID_PCM_S32LE;

            } else if (st->codec->codec_id == CODEC_ID_PCM_S16BE) {

                if (descriptor->bits_per_sample == 24)

                    st->codec->codec_id = CODEC_ID_PCM_S24BE;

                else if (descriptor->bits_per_sample == 32)

                    st->codec->codec_id = CODEC_ID_PCM_S32BE;

                if (descriptor->essence_container_ul[13] == 0x01) /* D-10 Mapping */

                    st->codec->channels = 8; /* force channels to 8 */

            } else if (st->codec->codec_id == CODEC_ID_MP2) {

                st->need_parsing = 1;

            }

        }

        if (container_ul && container_ul->wrapping == Clip) {

            dprintf("stream %d: clip wrapped essence\n", st->index);

            st->need_parsing = 1;

        }

    }

    return 0;

}
