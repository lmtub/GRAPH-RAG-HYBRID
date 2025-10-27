static gboolean vtd_hash_remove_by_page(gpointer key, gpointer value,

                                        gpointer user_data)

{

    VTDIOTLBEntry *entry = (VTDIOTLBEntry *)value;

    VTDIOTLBPageInvInfo *info = (VTDIOTLBPageInvInfo *)user_data;

    uint64_t gfn = info->gfn & info->mask;

    return (entry->domain_id == info->domain_id) &&

            ((entry->gfn & info->mask) == gfn);

}
