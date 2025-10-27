static void jpeg_term_destination(j_compress_ptr cinfo)

{

    VncState *vs = cinfo->client_data;

    Buffer *buffer = &vs->tight_jpeg;



    buffer->offset = buffer->capacity - cinfo->dest->free_in_buffer;

}
