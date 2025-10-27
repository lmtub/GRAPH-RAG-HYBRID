static int usage(int ret)

{

    fprintf(stderr, "dump (up to maxpkts) AVPackets as they are demuxed by libavformat.\n");

    fprintf(stderr, "each packet is dumped in its own file named like `basename file.ext`_$PKTNUM_$STREAMINDEX_$STAMP_$SIZE_$FLAGS.bin\n");

    fprintf(stderr, "pktdumper file [maxpkts]\n");

    return ret;

}
