static void set_fifodepth(MSSSpiState *s)

{

    unsigned int size = s->regs[R_SPI_DFSIZE] & FRAMESZ_MASK;



    if (size <= 8) {

        s->fifo_depth = 32;

    } else if (size <= 16) {

        s->fifo_depth = 16;

    } else if (size <= 32) {

        s->fifo_depth = 8;

    } else {

        s->fifo_depth = 4;

    }

}
