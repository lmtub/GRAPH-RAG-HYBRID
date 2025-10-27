static int aio_flush_f(BlockBackend *blk, int argc, char **argv)

{



    blk_drain_all();


    return 0;

}