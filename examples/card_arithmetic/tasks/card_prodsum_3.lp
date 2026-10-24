% Suit values
suit_value(d,1; c,2; s,3; h,4).

% Card value is the product of rank and suit values
card_value(P,RV*SV) :- suit(P,S), rank(P,R), rank_value(R,RV), suit_value(S,SV).

% Result is the sum of three cards
sum(0,V) :- card_value(0,V).
sum(P,V1+V2) :- sum(P-1,V1), card_value(P,V2).
result(X) :- sum(2,X).

% Neural predicate
nn(card(3,p), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]).
