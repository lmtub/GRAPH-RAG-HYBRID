void fifo_realloc(FifoBuffer *f, int new_size){

    int old_size= f->end - f->buffer;

    

    if(old_size < new_size){

        uint8_t *old= f->buffer;



        f->buffer= av_realloc(f->buffer, new_size);



        f->rptr += f->buffer - old;

        f->wptr += f->buffer - old;



        if(f->wptr < f->rptr){

            memmove(f->rptr + new_size - old_size, f->rptr, f->buffer + old_size - f->rptr);

            f->rptr += new_size - old_size;

        }

        f->end= f->buffer + new_size;

    }

}
