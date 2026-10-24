% Ranks and suits
rank(2..10). rank(a; j; q; k).
suit(h; s; d; c).

% Rank values
rank_value(R,R) :- rank(R), R = 2..10.
rank_value(j,11; q,12; k,13; a,14).

% Converting from cards to ranks and suits
rank(P,R) :- card(P,p,C), rank_value(R,C\13+2).
suit(P,h) :- card(P,p,C), C <= 12.
suit(P,c) :- card(P,p,C), C >= 13, C <= 25.
suit(P,s) :- card(P,p,C), C >= 26, C <= 38.
suit(P,d) :- card(P,p,C), C >= 39.