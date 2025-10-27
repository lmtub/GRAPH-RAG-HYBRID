static void buffer_append(Buffer *buffer, const void *data, size_t len)

{

    memcpy(buffer->buffer + buffer->offset, data, len);

    buffer->offset += len;

}
