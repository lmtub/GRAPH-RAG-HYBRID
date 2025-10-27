int qemu_acl_party_is_allowed(qemu_acl *acl,

                              const char *party)

{

    qemu_acl_entry *entry;



    TAILQ_FOREACH(entry, &acl->entries, next) {

#ifdef CONFIG_FNMATCH

        if (fnmatch(entry->match, party, 0) == 0)

            return entry->deny ? 0 : 1;

#else

        /* No fnmatch, so fallback to exact string matching

         * instead of allowing wildcards */

        if (strcmp(entry->match, party) == 0)

            return entry->deny ? 0 : 1;

#endif

    }



    return acl->defaultDeny ? 0 : 1;

}
