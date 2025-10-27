static void aio_signal_handler(int signum)

{

    if (posix_aio_state) {

        char byte = 0;



        write(posix_aio_state->wfd, &byte, sizeof(byte));

    }



    qemu_service_io();

}
