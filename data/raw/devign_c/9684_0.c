void qemu_acl_reset(qemu_acl *acl)

{

    qemu_acl_entry *entry;



    /* Put back to deny by default, so there is no window

     * of "open access" while the user re-initializes the

     * access control list */

    acl->defaultDeny = 1;

    TAILQ_FOREACH(entry, &acl->entries, next) {

        TAILQ_REMOVE(&acl->entries, entry, next);

        free(entry->match);

        free(entry);

    }

    acl->nentries = 0;

}
