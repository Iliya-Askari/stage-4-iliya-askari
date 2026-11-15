# One-page summary for the Head of Operations

## What the system does
Every time an order is confirmed, the system gives it a score for how likely the customer is to end up unhappy and leave a bad
review. It then makes a list of the orders that deserve a call or a message first. The list is sized to what your team can handle:
the top 10 orders out of every 100.

## How much can you rely on it?
Think of 100 customers who will end up unhappy.
* Contacting 10 of every 100 orders at random reaches about 10 of them.
* A simple rule ("call the customers of sellers with the worst reputation") reaches about 17.
* This list reaches about **24**. In the most recent month of data it reached about 21.
* If your team can handle more, contacting 20 of every 100 orders reaches about 37 of the 100; contacting only 5 reaches about 16.

Of the orders on the list, about 27 in 100 really end badly, against about 12 in 100 across all orders. So the list is a good
way to decide who to look at first, but it is not a crystal ball: most listed orders would have been fine, and about three out of
four unhappy customers will not be on the list.

The system only uses facts that are really known when an order is confirmed. We also tested what happens if it were allowed to
peek at facts that are only known later (for example whether the parcel arrived late): it would then look almost perfect on
paper, catching more than half of the unhappy customers, but it would be useless in real life. We have made sure this cannot
happen by accident.

## What your team should do
1. Each day (or each batch), work through the list from the top until the capacity is used.
2. Keep a note of who was contacted and what happened, so the next review can measure whether the calls help.
3. Once a month, check how many of the customers who turned out unhappy had been on the list. Today it is about one in four to
   one in five. If it drops clearly below one in six, ask the data team to look at the system.
4. Ask the data team to refresh the system whenever delivery promises, carriers or the seller mix change noticeably: the platform
   changed during the data period and the list is a little less sharp in recent months.
5. Do not use the list to judge individual sellers or staff; it is a priority guide.
