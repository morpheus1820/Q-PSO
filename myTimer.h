/*
 * myTimer.h
 *
 *  Created on: 02/gen/2016
 *      Author: giorgio
 */

#ifndef MYTIMER_H_
#define MYTIMER_H_

#include <sys/time.h>
#include <stdio.h>
#include <string>

class Timer
{
public:
    static const uint8_t MILLI=0;
    static const uint8_t MICRO=1;
    static const uint8_t SEC=2;
    Timer()
    {
    	startCount.tv_sec = startCount.tv_usec = 0;
    	endCount.tv_sec = endCount.tv_usec = 0;

    	stopped = 0;
    	startTimeInMicroSec = 0;
    	endTimeInMicroSec = 0;
    }
    ~Timer(){}
    inline void start()
    {
        stopped = 0; // reset stop flag
        gettimeofday(&startCount, NULL);
    }
    inline void stop()
    {
        gettimeofday(&endCount, NULL);
        stopped = 1; // set timer stopped flag
    }
    inline double getElapsedTime(const uint8_t unit_,std::string name)
    {

        if(!stopped)
        {
            gettimeofday(&endCount, NULL);
            stopped = 1;
        }

        startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
        endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;

        double Diff = endTimeInMicroSec - startTimeInMicroSec;

        switch (unit_) {
			case MICRO:
				name += ": %f usec\n";
				printf(name.c_str(),Diff);
				return Diff;
			case MILLI:default:
				name += ": %f msec\n";
				Diff *= 0.001;
				printf(name.c_str(),Diff);
				return Diff;
			case SEC:
				name += ": %f sec\n";
				Diff *= 0.000001;
				printf(name.c_str(),Diff);
				return Diff;
		}


    }

private:
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int    stopped;                             // stop flag

    timeval startCount;                         //
    timeval endCount;                           //
};


#endif /* MYTIMER_H_ */
