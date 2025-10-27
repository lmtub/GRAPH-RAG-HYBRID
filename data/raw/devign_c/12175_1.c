static void bt_l2cap_sdp_close_ch(void *opaque)

{

    struct bt_l2cap_sdp_state_s *sdp = opaque;

    int i;



    for (i = 0; i < sdp->services; i ++) {

        g_free(sdp->service_list[i].attribute_list->pair);

        g_free(sdp->service_list[i].attribute_list);

        g_free(sdp->service_list[i].uuid);

    }

    g_free(sdp->service_list);

    g_free(sdp);

}
