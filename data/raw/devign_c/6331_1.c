static void buffer_reserve(Buffer *buffer, size_t len)

{

    if ((buffer->capacity - buffer->offset) < len) {

	buffer->capacity += (len + 1024);

	buffer->buffer = qemu_realloc(buffer->buffer, buffer->capacity);

	if (buffer->buffer == NULL) {

	    fprintf(stderr, "vnc: out of memory\n");

	    exit(1);

	}

    }

}
