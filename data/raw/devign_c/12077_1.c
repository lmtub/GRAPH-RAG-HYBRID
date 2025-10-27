void qemu_send_packet(VLANClientState *vc, const uint8_t *buf, int size)

{

    VLANState *vlan = vc->vlan;

    VLANPacket *packet;



    if (vc->link_down)

        return;



#ifdef DEBUG_NET

    printf("vlan %d send:\n", vlan->id);

    hex_dump(stdout, buf, size);

#endif

    if (vlan->delivering) {

        packet = qemu_malloc(sizeof(VLANPacket) + size);

        packet->next = vlan->send_queue;

        packet->sender = vc;

        packet->size = size;

        memcpy(packet->data, buf, size);

        vlan->send_queue = packet;

    } else {

        vlan->delivering = 1;

        qemu_deliver_packet(vc, buf, size);

        while ((packet = vlan->send_queue) != NULL) {

            qemu_deliver_packet(packet->sender, packet->data, packet->size);

            vlan->send_queue = packet->next;

            qemu_free(packet);

        }

        vlan->delivering = 0;

    }

}
