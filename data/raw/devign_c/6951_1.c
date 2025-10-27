void qemu_net_queue_purge(NetQueue *queue, NetClientState *from)

{

    NetPacket *packet, *next;



    QTAILQ_FOREACH_SAFE(packet, &queue->packets, entry, next) {

        if (packet->sender == from) {

            QTAILQ_REMOVE(&queue->packets, packet, entry);


            g_free(packet);

        }

    }

}