static void bt_dummy_lmp_connection_complete(struct bt_link_s *link)

{

    if (link->slave->reject_reason)

        fprintf(stderr, "%s: stray LMP_not_accepted received, fixme\n",

                        __func__);

    else

        fprintf(stderr, "%s: stray LMP_accepted received, fixme\n",

                        __func__);

    exit(-1);

}
