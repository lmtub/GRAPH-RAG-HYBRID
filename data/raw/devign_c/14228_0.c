void qemu_flush_queued_packets(VLANClientState *vc)

{

    while (!TAILQ_EMPTY(&vc->vlan->send_queue)) {

        VLANPacket *packet;

        int ret;



        packet = TAILQ_FIRST(&vc->vlan->send_queue);

        TAILQ_REMOVE(&vc->vlan->send_queue, packet, entry);



        ret = qemu_deliver_packet(packet->sender, packet->data, packet->size);

        if (ret == 0 && packet->sent_cb != NULL) {

            TAILQ_INSERT_HEAD(&vc->vlan->send_queue, packet, entry);

            break;

        }



        if (packet->sent_cb)

            packet->sent_cb(packet->sender, ret);



        qemu_free(packet);

    }

}
