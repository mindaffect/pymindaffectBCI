package nl.ma.utopiaserver;
/** 
 * Exception class for when a client message is somehow malformed or unparseable.
 */
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */
public class ClientException extends Exception {
	public ClientException(final String string) {
		super(string);
	}
}
