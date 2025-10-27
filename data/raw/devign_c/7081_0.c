static boolean jpeg_empty_output_buffer(j_compress_ptr cinfo)

{

    VncState *vs = cinfo->client_data;

    Buffer *buffer = &vs->tight_jpeg;



    buffer->offset = buffer->capacity;

    buffer_reserve(buffer, 2048);

    jpeg_init_destination(cinfo);

    return TRUE;

}
