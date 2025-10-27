static void destroy_buffers(VADisplay display, VABufferID *buffers, unsigned int n_buffers)

{

    unsigned int i;

    for (i = 0; i < n_buffers; i++) {

        if (buffers[i]) {

            vaDestroyBuffer(display, buffers[i]);

            buffers[i] = 0;

        }

    }

}
