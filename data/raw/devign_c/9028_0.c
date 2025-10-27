GuestLogicalProcessorList *qmp_guest_get_vcpus(Error **errp)

{

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION pslpi, ptr;

    DWORD length;

    GuestLogicalProcessorList *head, **link;

    Error *local_err = NULL;

    int64_t current;



    ptr = pslpi = NULL;

    length = 0;

    current = 0;

    head = NULL;

    link = &head;



    if ((GetLogicalProcessorInformation(pslpi, &length) == FALSE) &&

        (GetLastError() == ERROR_INSUFFICIENT_BUFFER) &&

        (length > sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION))) {

        ptr = pslpi = g_malloc0(length);

        if (GetLogicalProcessorInformation(pslpi, &length) == FALSE) {

            error_setg(&local_err, "Failed to get processor information: %d",

                       (int)GetLastError());

        }

    } else {

        error_setg(&local_err,

                   "Failed to get processor information buffer length: %d",

                   (int)GetLastError());

    }



    while ((local_err == NULL) && (length > 0)) {

        if (pslpi->Relationship == RelationProcessorCore) {

            ULONG_PTR cpu_bits = pslpi->ProcessorMask;



            while (cpu_bits > 0) {

                if (!!(cpu_bits & 1)) {

                    GuestLogicalProcessor *vcpu;

                    GuestLogicalProcessorList *entry;



                    vcpu = g_malloc0(sizeof *vcpu);

                    vcpu->logical_id = current++;

                    vcpu->online = true;

                    vcpu->has_can_offline = false;



                    entry = g_malloc0(sizeof *entry);

                    entry->value = vcpu;



                    *link = entry;

                    link = &entry->next;

                }

                cpu_bits >>= 1;

            }

        }

        length -= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);

        pslpi++; /* next entry */

    }



    g_free(ptr);



    if (local_err == NULL) {

        if (head != NULL) {

            return head;

        }

        /* there's no guest with zero VCPUs */

        error_setg(&local_err, "Guest reported zero VCPUs");

    }



    qapi_free_GuestLogicalProcessorList(head);

    error_propagate(errp, local_err);

    return NULL;

}
