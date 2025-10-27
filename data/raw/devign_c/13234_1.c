strdup(str)

	const char *str;

{

	char *bptr;



	bptr = (char *)malloc(strlen(str)+1);

	strcpy(bptr, str);



	return bptr;

}
