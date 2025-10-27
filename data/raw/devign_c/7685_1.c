static void print_net_client(Monitor *mon, VLANClientState *vc)

{

    monitor_printf(mon, "%s: type=%s,%s\n", vc->name,

                   net_client_types[vc->info->type].type, vc->info_str);

}
