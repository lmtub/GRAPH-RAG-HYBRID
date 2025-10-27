static int buffered_put_buffer(void *opaque, const uint8_t *buf, int64_t pos, int size)

{

    QEMUFileBuffered *s = opaque;

    ssize_t error;



    DPRINTF("putting %d bytes at %" PRId64 "\n", size, pos);



    error = qemu_file_get_error(s->file);

    if (error) {

        DPRINTF("flush when error, bailing: %s\n", strerror(-error));

        return error;

    }



    if (size <= 0) {

        return size;

    }



    if (size > (s->buffer_capacity - s->buffer_size)) {

        DPRINTF("increasing buffer capacity from %zu by %zu\n",

                s->buffer_capacity, size + 1024);



        s->buffer_capacity += size + 1024;



        s->buffer = g_realloc(s->buffer, s->buffer_capacity);

    }



    memcpy(s->buffer + s->buffer_size, buf, size);

    s->buffer_size += size;



    return size;

}
