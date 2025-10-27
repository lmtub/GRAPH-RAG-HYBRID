static void input_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)

{

    if (!buffer->cmd) {

        AVBufferRef *buf = buffer->user_data;

        av_buffer_unref(&buf);

    }

    mmal_buffer_header_release(buffer);

}
