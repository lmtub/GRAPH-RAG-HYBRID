envlist_create(void)

{

	envlist_t *envlist;



	if ((envlist = malloc(sizeof (*envlist))) == NULL)

		return (NULL);



	QLIST_INIT(&envlist->el_entries);

	envlist->el_count = 0;



	return (envlist);

}
