static void bt_dummy_lmp_acl_resp(struct bt_link_s *link,

                const uint8_t *data, int start, int len)

{

    fprintf(stderr, "%s: stray ACL response PDU, fixme\n", __FUNCTION__);

    exit(-1);

}
