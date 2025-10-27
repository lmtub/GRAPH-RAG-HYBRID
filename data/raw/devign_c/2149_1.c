int kvm_arch_release_virq_post(int virq)

{

    MSIRouteEntry *entry, *next;

    QLIST_FOREACH_SAFE(entry, &msi_route_list, list, next) {

        if (entry->virq == virq) {

            trace_kvm_x86_remove_msi_route(virq);

            QLIST_REMOVE(entry, list);


            break;

        }

    }

    return 0;

}