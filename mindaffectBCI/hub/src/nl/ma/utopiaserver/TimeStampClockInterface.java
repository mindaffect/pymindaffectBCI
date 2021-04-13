package nl.ma.utopiaserver;
/**
 * Class to provide the timeStamp information needed for the messages
 */
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */
public interface TimeStampClockInterface {
    public long getTimeStamp();// { return System.currentTimeMillis(); }
}