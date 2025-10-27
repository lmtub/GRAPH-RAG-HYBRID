static void control_out(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtQueueElement elem;

    VirtIOSerial *vser;



    vser = DO_UPCAST(VirtIOSerial, vdev, vdev);



    while (virtqueue_pop(vq, &elem)) {

        handle_control_message(vser, elem.out_sg[0].iov_base);

        virtqueue_push(vq, &elem, elem.out_sg[0].iov_len);

    }

    virtio_notify(vdev, vq);

}
