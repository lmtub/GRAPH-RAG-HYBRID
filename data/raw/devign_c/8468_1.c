envlist_to_environ(const envlist_t *envlist, size_t *count)

{

	struct envlist_entry *entry;

	char **env, **penv;



	penv = env = malloc((envlist->el_count + 1) * sizeof (char *));

	if (env == NULL)

		return (NULL);



	for (entry = envlist->el_entries.lh_first; entry != NULL;

	    entry = entry->ev_link.le_next) {

		*(penv++) = strdup(entry->ev_var);

	}

	*penv = NULL; /* NULL terminate the list */



	if (count != NULL)

		*count = envlist->el_count;



	return (env);

}
