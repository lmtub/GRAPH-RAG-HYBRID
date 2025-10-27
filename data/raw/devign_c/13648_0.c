static void unref_buffer(InputStream *ist, FrameBuffer *buf)

{

    av_assert0(buf->refcount);

    buf->refcount--;

    if (!buf->refcount) {

        buf->next = ist->buffer_pool;

        ist->buffer_pool = buf;

    }

}
