static void bt_dummy_lmp_disconnect_master(struct bt_link_s *link)

{

    fprintf(stderr, "%s: stray LMP_detach received, fixme\n", __func__);

    exit(-1);

}
