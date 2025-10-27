static void colo_old_packet_check(void *opaque)

{

    CompareState *s = opaque;



    g_queue_foreach(&s->conn_list, colo_old_packet_check_one_conn, NULL);

}
