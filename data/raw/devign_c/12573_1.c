static void hmp_info_cpustats(Monitor *mon, const QDict *qdict)

{

    cpu_dump_statistics(mon_get_cpu(), (FILE *)mon, &monitor_fprintf, 0);

}
