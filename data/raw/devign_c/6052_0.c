fetchline(void)

{

	char	*p, *line = malloc(MAXREADLINESZ);



	if (!line)

		return NULL;

	printf("%s", get_prompt());

	fflush(stdout);

	if (!fgets(line, MAXREADLINESZ, stdin)) {

		free(line);

		return NULL;

	}

	p = line + strlen(line);

	if (p != line && p[-1] == '\n')

		p[-1] = '\0';

	return line;

}
