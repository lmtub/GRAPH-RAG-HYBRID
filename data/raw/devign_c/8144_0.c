static void colo_compare_connection(void *opaque, void *user_data)

{

    CompareState *s = user_data;

    Connection *conn = opaque;

    Packet *pkt = NULL;

    GList *result = NULL;

    int ret;



    while (!g_queue_is_empty(&conn->primary_list) &&

           !g_queue_is_empty(&conn->secondary_list)) {

        qemu_mutex_lock(&s->timer_check_lock);

        pkt = g_queue_pop_tail(&conn->primary_list);

        qemu_mutex_unlock(&s->timer_check_lock);

        result = g_queue_find_custom(&conn->secondary_list,

                              pkt, (GCompareFunc)colo_packet_compare_all);



        if (result) {

            ret = compare_chr_send(s->chr_out, pkt->data, pkt->size);

            if (ret < 0) {

                error_report("colo_send_primary_packet failed");

            }

            trace_colo_compare_main("packet same and release packet");

            g_queue_remove(&conn->secondary_list, result->data);

            packet_destroy(pkt, NULL);

        } else {

            /*

             * If one packet arrive late, the secondary_list or

             * primary_list will be empty, so we can't compare it

             * until next comparison.

             */

            trace_colo_compare_main("packet different");

            qemu_mutex_lock(&s->timer_check_lock);

            g_queue_push_tail(&conn->primary_list, pkt);

            qemu_mutex_unlock(&s->timer_check_lock);

            /* TODO: colo_notify_checkpoint();*/

            break;

        }

    }

}
