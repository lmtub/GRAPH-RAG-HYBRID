void ff_nut_free_sp(NUTContext *nut)

{

    av_tree_enumerate(nut->syncpoints, NULL, NULL, enu_free);

    av_tree_destroy(nut->syncpoints);

}
