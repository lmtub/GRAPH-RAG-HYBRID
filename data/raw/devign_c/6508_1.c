static void do_flush_queued_data(VirtIOSerialPort *port, VirtQueue *vq,

                                 VirtIODevice *vdev)

{

    VirtIOSerialPortClass *vsc;



    assert(port);

    assert(virtio_queue_ready(vq));



    vsc = VIRTIO_SERIAL_PORT_GET_CLASS(port);



    while (!port->throttled) {

        unsigned int i;



        /* Pop an elem only if we haven't left off a previous one mid-way */

        if (!port->elem) {

            port->elem = virtqueue_pop(vq, sizeof(VirtQueueElement));

            if (!port->elem) {

                break;


            port->iov_idx = 0;

            port->iov_offset = 0;




        for (i = port->iov_idx; i < port->elem->out_num; i++) {

            size_t buf_size;

            ssize_t ret;



            buf_size = port->elem->out_sg[i].iov_len - port->iov_offset;

            ret = vsc->have_data(port,

                                  port->elem->out_sg[i].iov_base

                                  + port->iov_offset,

                                  buf_size);




            if (port->throttled) {

                port->iov_idx = i;

                if (ret > 0) {

                    port->iov_offset += ret;


                break;


            port->iov_offset = 0;


        if (port->throttled) {

            break;


        virtqueue_push(vq, port->elem, 0);

        g_free(port->elem);

        port->elem = NULL;


    virtio_notify(vdev, vq);
