static int openfile(char *name, int flags, int growable)

{

	if (bs) {

		fprintf(stderr, "file open already, try 'help close'\n");

		return 1;

	}



	bs = bdrv_new("hda");

	if (!bs)

		return 1;



	if (bdrv_open(bs, name, flags) == -1) {

		fprintf(stderr, "%s: can't open device %s\n", progname, name);

		bs = NULL;

		return 1;

	}





	if (growable) {

		if (!bs->drv || !bs->drv->protocol_name) {

			fprintf(stderr,

				"%s: only protocols can be opened growable\n",

				progname);

			return 1;

		}

		bs->growable = 1;

	}



	return 0;

}
