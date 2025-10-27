void av_register_all(void)

{

    static int initialized;



    if (initialized)

        return;

    initialized = 1;



    avcodec_init();

    avcodec_register_all();



    /* (de)muxers */

    REGISTER_DEMUXER  (AAC, aac);

    REGISTER_MUXDEMUX (AC3, ac3);

    REGISTER_MUXER    (ADTS, adts);

    REGISTER_MUXDEMUX (AIFF, aiff);

    REGISTER_MUXDEMUX (AMR, amr);

    REGISTER_DEMUXER  (APC, apc);

    REGISTER_DEMUXER  (APE, ape);

    REGISTER_MUXDEMUX (ASF, asf);

    REGISTER_MUXER    (ASF_STREAM, asf_stream);

    REGISTER_MUXDEMUX (AU, au);

    REGISTER_MUXDEMUX (AVI, avi);

    REGISTER_DEMUXER  (AVISYNTH, avisynth);

    REGISTER_MUXER    (AVM2, avm2);

    REGISTER_DEMUXER  (AVS, avs);

    REGISTER_DEMUXER  (BETHSOFTVID, bethsoftvid);

    REGISTER_DEMUXER  (BFI, bfi);

    REGISTER_DEMUXER  (C93, c93);

    REGISTER_MUXER    (CRC, crc);

    REGISTER_DEMUXER  (DAUD, daud);

    REGISTER_MUXDEMUX (DIRAC, dirac);

    REGISTER_DEMUXER  (DSICIN, dsicin);

    REGISTER_MUXDEMUX (DTS, dts);

    REGISTER_MUXDEMUX (DV, dv);

    REGISTER_DEMUXER  (DXA, dxa);

    REGISTER_DEMUXER  (EA, ea);

    REGISTER_DEMUXER  (EA_CDATA, ea_cdata);

    REGISTER_MUXDEMUX (FFM, ffm);

    REGISTER_MUXDEMUX (FLAC, flac);

    REGISTER_DEMUXER  (FLIC, flic);

    REGISTER_MUXDEMUX (FLV, flv);

    REGISTER_DEMUXER  (FOURXM, fourxm);

    REGISTER_MUXER    (FRAMECRC, framecrc);

    REGISTER_MUXDEMUX (GIF, gif);

    REGISTER_DEMUXER  (GSM, gsm);

    REGISTER_MUXDEMUX (GXF, gxf);

    REGISTER_MUXDEMUX (H261, h261);

    REGISTER_MUXDEMUX (H263, h263);

    REGISTER_MUXDEMUX (H264, h264);

    REGISTER_DEMUXER  (IDCIN, idcin);

    REGISTER_DEMUXER  (IFF, iff);

    REGISTER_MUXDEMUX (IMAGE2, image2);

    REGISTER_MUXDEMUX (IMAGE2PIPE, image2pipe);

    REGISTER_DEMUXER  (INGENIENT, ingenient);

    REGISTER_DEMUXER  (IPMOVIE, ipmovie);

    REGISTER_MUXER    (IPOD, ipod);

    REGISTER_DEMUXER  (LMLM4, lmlm4);

    REGISTER_MUXDEMUX (M4V, m4v);

    REGISTER_MUXDEMUX (MATROSKA, matroska);

    REGISTER_MUXER    (MATROSKA_AUDIO, matroska_audio);

    REGISTER_MUXDEMUX (MJPEG, mjpeg);

    REGISTER_DEMUXER  (MLP, mlp);

    REGISTER_DEMUXER  (MM, mm);

    REGISTER_MUXDEMUX (MMF, mmf);

    REGISTER_MUXDEMUX (MOV, mov);

    REGISTER_MUXER    (MP2, mp2);

    REGISTER_MUXDEMUX (MP3, mp3);

    REGISTER_MUXER    (MP4, mp4);

    REGISTER_DEMUXER  (MPC, mpc);

    REGISTER_DEMUXER  (MPC8, mpc8);

    REGISTER_MUXER    (MPEG1SYSTEM, mpeg1system);

    REGISTER_MUXER    (MPEG1VCD, mpeg1vcd);

    REGISTER_MUXER    (MPEG1VIDEO, mpeg1video);

    REGISTER_MUXER    (MPEG2DVD, mpeg2dvd);

    REGISTER_MUXER    (MPEG2SVCD, mpeg2svcd);

    REGISTER_MUXER    (MPEG2VIDEO, mpeg2video);

    REGISTER_MUXER    (MPEG2VOB, mpeg2vob);

    REGISTER_DEMUXER  (MPEGPS, mpegps);

    REGISTER_MUXDEMUX (MPEGTS, mpegts);

    REGISTER_DEMUXER  (MPEGTSRAW, mpegtsraw);

    REGISTER_DEMUXER  (MPEGVIDEO, mpegvideo);

    REGISTER_MUXER    (MPJPEG, mpjpeg);

    REGISTER_DEMUXER  (MSNWC_TCP, msnwc_tcp);

    REGISTER_DEMUXER  (MTV, mtv);

    REGISTER_DEMUXER  (MVI, mvi);

    REGISTER_DEMUXER  (MXF, mxf);

    REGISTER_DEMUXER  (NSV, nsv);

    REGISTER_MUXER    (NULL, null);

    REGISTER_MUXDEMUX (NUT, nut);

    REGISTER_DEMUXER  (NUV, nuv);

    REGISTER_MUXDEMUX (OGG, ogg);

    REGISTER_DEMUXER  (OMA, oma);

    REGISTER_MUXDEMUX (PCM_ALAW,  pcm_alaw);

    REGISTER_MUXDEMUX (PCM_MULAW, pcm_mulaw);

    REGISTER_MUXDEMUX (PCM_S16BE, pcm_s16be);

    REGISTER_MUXDEMUX (PCM_S16LE, pcm_s16le);

    REGISTER_MUXDEMUX (PCM_S8,    pcm_s8);

    REGISTER_MUXDEMUX (PCM_U16BE, pcm_u16be);

    REGISTER_MUXDEMUX (PCM_U16LE, pcm_u16le);

    REGISTER_MUXDEMUX (PCM_U8,    pcm_u8);

    REGISTER_MUXER    (PSP, psp);

    REGISTER_DEMUXER  (PVA, pva);

    REGISTER_MUXDEMUX (RAWVIDEO, rawvideo);

    REGISTER_DEMUXER  (REDIR, redir);

    REGISTER_DEMUXER  (RL2, rl2);

    REGISTER_MUXDEMUX (RM, rm);

    REGISTER_MUXDEMUX (ROQ, roq);

    REGISTER_DEMUXER  (RPL, rpl);

    REGISTER_MUXER    (RTP, rtp);

    REGISTER_DEMUXER  (RTSP, rtsp);

    REGISTER_DEMUXER  (SDP, sdp);

#ifdef CONFIG_SDP_DEMUXER

    av_register_rtp_dynamic_payload_handlers();

#endif

    REGISTER_DEMUXER  (SEGAFILM, segafilm);

    REGISTER_DEMUXER  (SHORTEN, shorten);

    REGISTER_DEMUXER  (SIFF, siff);

    REGISTER_DEMUXER  (SMACKER, smacker);

    REGISTER_DEMUXER  (SOL, sol);

    REGISTER_DEMUXER  (STR, str);

    REGISTER_MUXDEMUX (SWF, swf);

    REGISTER_MUXER    (TG2, tg2);

    REGISTER_MUXER    (TGP, tgp);

    REGISTER_DEMUXER  (THP, thp);

    REGISTER_DEMUXER  (TIERTEXSEQ, tiertexseq);

    REGISTER_DEMUXER  (TTA, tta);

    REGISTER_DEMUXER  (TXD, txd);

    REGISTER_DEMUXER  (VC1, vc1);

    REGISTER_DEMUXER  (VC1T, vc1t);

    REGISTER_DEMUXER  (VMD, vmd);

    REGISTER_MUXDEMUX (VOC, voc);

    REGISTER_MUXDEMUX (WAV, wav);

    REGISTER_DEMUXER  (WC3, wc3);

    REGISTER_DEMUXER  (WSAUD, wsaud);

    REGISTER_DEMUXER  (WSVQA, wsvqa);

    REGISTER_DEMUXER  (WV, wv);

    REGISTER_DEMUXER  (XA, xa);

    REGISTER_MUXDEMUX (YUV4MPEGPIPE, yuv4mpegpipe);



    /* external libraries */

    REGISTER_MUXDEMUX (LIBNUT, libnut);



    /* protocols */

    REGISTER_PROTOCOL (FILE, file);

    REGISTER_PROTOCOL (HTTP, http);

    REGISTER_PROTOCOL (PIPE, pipe);

    REGISTER_PROTOCOL (RTP, rtp);

    REGISTER_PROTOCOL (TCP, tcp);

    REGISTER_PROTOCOL (UDP, udp);

}
