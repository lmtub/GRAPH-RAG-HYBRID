static int packet_enqueue(CompareState *s, int mode)

{

    ConnectionKey key;

    Packet *pkt = NULL;

    Connection *conn;



    if (mode == PRIMARY_IN) {

        pkt = packet_new(s->pri_rs.buf,

                         s->pri_rs.packet_len,

                         s->pri_rs.vnet_hdr_len);

    } else {

        pkt = packet_new(s->sec_rs.buf,

                         s->sec_rs.packet_len,

                         s->sec_rs.vnet_hdr_len);

    }



    if (parse_packet_early(pkt)) {

        packet_destroy(pkt, NULL);

        pkt = NULL;

        return -1;

    }

    fill_connection_key(pkt, &key);



    conn = connection_get(s->connection_track_table,

                          &key,

                          &s->conn_list);



    if (!conn->processing) {

        g_queue_push_tail(&s->conn_list, conn);

        conn->processing = true;

    }



    if (mode == PRIMARY_IN) {

        if (!colo_insert_packet(&conn->primary_list, pkt)) {

            error_report("colo compare primary queue size too big,"

                         "drop packet");

        }

    } else {

        if (!colo_insert_packet(&conn->secondary_list, pkt)) {

            error_report("colo compare secondary queue size too big,"

                         "drop packet");

        }

    }



    return 0;

}
