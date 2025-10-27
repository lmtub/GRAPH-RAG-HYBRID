int av_fifo_generic_write(AVFifoBuffer *f, void *src, int size, int (*func)(void*, void*, int))

{

    int total = size;

    do {

        int len = FFMIN(f->end - f->wptr, size);

        if (func) {

            if (func(src, f->wptr, len) <= 0)

                break;

        } else {

            memcpy(f->wptr, src, len);

            src = (uint8_t*)src + len;

        }

// Write memory barrier needed for SMP here in theory

        f->wptr += len;

        if (f->wptr >= f->end)

            f->wptr = f->buffer;

        f->wndx += len;

        size -= len;

    } while (size > 0);

    return total - size;

}
