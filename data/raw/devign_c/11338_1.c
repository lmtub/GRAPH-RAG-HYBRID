static int iec61883_read_header(AVFormatContext *context)

{

    struct iec61883_data *dv = context->priv_data;

    struct raw1394_portinfo pinf[16];

    rom1394_directory rom_dir;

    char *endptr;

    int inport;

    int nb_ports;

    int port = -1;

    int response;

    int i, j = 0;

    uint64_t guid = 0;



    dv->input_port = -1;

    dv->output_port = -1;

    dv->channel = -1;



    dv->raw1394 = raw1394_new_handle();



    if (!dv->raw1394) {

        av_log(context, AV_LOG_ERROR, "Failed to open IEEE1394 interface.\n");

        return AVERROR(EIO);

    }



    if ((nb_ports = raw1394_get_port_info(dv->raw1394, pinf, 16)) < 0) {

        av_log(context, AV_LOG_ERROR, "Failed to get number of IEEE1394 ports.\n");

        goto fail;

    }



    inport = strtol(context->filename, &endptr, 10);

    if (endptr != context->filename && *endptr == '\0') {

        av_log(context, AV_LOG_INFO, "Selecting IEEE1394 port: %d\n", inport);

        j = inport;

        nb_ports = inport + 1;

    } else if (strcmp(context->filename, "auto")) {

        av_log(context, AV_LOG_ERROR, "Invalid input \"%s\", you should specify "

               "\"auto\" for auto-detection, or the port number.\n", context->filename);

        goto fail;

    }



    if (dv->device_guid) {

        if (sscanf(dv->device_guid, "%llx", (long long unsigned int *)&guid) != 1) {

            av_log(context, AV_LOG_INFO, "Invalid dvguid parameter: %s\n",

                   dv->device_guid);

            goto fail;

        }

    }



    for (; j < nb_ports && port==-1; ++j) {

        raw1394_destroy_handle(dv->raw1394);



        if (!(dv->raw1394 = raw1394_new_handle_on_port(j))) {

            av_log(context, AV_LOG_ERROR, "Failed setting IEEE1394 port.\n");

            goto fail;

        }



        for (i=0; i<raw1394_get_nodecount(dv->raw1394); ++i) {



            /* Select device explicitly by GUID */



            if (guid > 1) {

                if (guid == rom1394_get_guid(dv->raw1394, i)) {

                    dv->node = i;

                    port = j;

                    break;

                }

            } else {



                /* Select first AV/C tape recorder player node */



                if (rom1394_get_directory(dv->raw1394, i, &rom_dir) < 0)

                    continue;

                if (((rom1394_get_node_type(&rom_dir) == ROM1394_NODE_TYPE_AVC) &&

                     avc1394_check_subunit_type(dv->raw1394, i, AVC1394_SUBUNIT_TYPE_VCR)) ||

                    (rom_dir.unit_spec_id == MOTDCT_SPEC_ID)) {

                    rom1394_free_directory(&rom_dir);

                    dv->node = i;

                    port = j;

                    break;

                }

                rom1394_free_directory(&rom_dir);

            }

        }

    }



    if (port == -1) {

        av_log(context, AV_LOG_ERROR, "No AV/C devices found.\n");

        goto fail;

    }



    /* Provide bus sanity for multiple connections */



    iec61883_cmp_normalize_output(dv->raw1394, 0xffc0 | dv->node);



    /* Find out if device is DV or HDV */



    if (dv->type == IEC61883_AUTO) {

        response = avc1394_transaction(dv->raw1394, dv->node,

                                       AVC1394_CTYPE_STATUS |

                                       AVC1394_SUBUNIT_TYPE_TAPE_RECORDER |

                                       AVC1394_SUBUNIT_ID_0 |

                                       AVC1394_VCR_COMMAND_OUTPUT_SIGNAL_MODE |

                                       0xFF, 2);

        response = AVC1394_GET_OPERAND0(response);

        dv->type = (response == 0x10 || response == 0x90 || response == 0x1A || response == 0x9A) ?

            IEC61883_HDV : IEC61883_DV;

    }



    /* Connect to device, and do initialization */



    dv->channel = iec61883_cmp_connect(dv->raw1394, dv->node, &dv->output_port,

                                       raw1394_get_local_id(dv->raw1394),

                                       &dv->input_port, &dv->bandwidth);



    if (dv->channel < 0)

        dv->channel = 63;



    if (!dv->max_packets)

        dv->max_packets = 100;



    if (CONFIG_MPEGTS_DEMUXER && dv->type == IEC61883_HDV) {



        /* Init HDV receive */



        avformat_new_stream(context, NULL);



        dv->mpeg_demux = avpriv_mpegts_parse_open(context);

        if (!dv->mpeg_demux)

            goto fail;



        dv->parse_queue = iec61883_parse_queue_hdv;



        dv->iec61883_mpeg2 = iec61883_mpeg2_recv_init(dv->raw1394,

                                                      (iec61883_mpeg2_recv_t)iec61883_callback,

                                                      dv);



        dv->max_packets *= 766;

    } else {



        /* Init DV receive */



        dv->dv_demux = avpriv_dv_init_demux(context);

        if (!dv->dv_demux)

            goto fail;



        dv->parse_queue = iec61883_parse_queue_dv;



        dv->iec61883_dv = iec61883_dv_fb_init(dv->raw1394, iec61883_callback, dv);

    }



    dv->raw1394_poll.fd = raw1394_get_fd(dv->raw1394);

    dv->raw1394_poll.events = POLLIN | POLLERR | POLLHUP | POLLPRI;



    /* Actually start receiving */



    if (dv->type == IEC61883_HDV)

        iec61883_mpeg2_recv_start(dv->iec61883_mpeg2, dv->channel);

    else

        iec61883_dv_fb_start(dv->iec61883_dv, dv->channel);



#if THREADS

    dv->thread_loop = 1;

    pthread_mutex_init(&dv->mutex, NULL);

    pthread_cond_init(&dv->cond, NULL);

    pthread_create(&dv->receive_task_thread, NULL, iec61883_receive_task, dv);

#endif



    return 0;



fail:

    raw1394_destroy_handle(dv->raw1394);

    return AVERROR(EIO);

}
