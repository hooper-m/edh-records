import datetime
import json
import statistics
from collections import Counter

from trueskill import Rating, rate


class Deck:
    def __init__(self, name, architect):
        self.name = name + ' - ' + architect
        self.simple_name = name
        self.architect = architect
        self.wins = 0
        self.played = 0
        self.score = 0
        self.matchups = {}
        self.wins_by_turn = Counter()
        self.eliminations = Counter()
        self.mmr = Rating()
        self._assists = 0
        self.games = []
        self.ranks = {
            'by_tbs': '',
            'by_delta': '',
            'by_wrx1': '',
            'by_mmr': '',
            'by_speed': ''
        }
        self.rank_delta = {
            'by_tbs': '',
            'by_delta': '',
            'by_wrx1': '',
            'by_mmr': '',
            'by_speed': ''
        }
        self.winrate = None
        self.expected_winrate = None
        self.winrate_delta = None
        self.op_winrate = None
        self.wrx1 = None
        self.mu = None
        self.sigma = None

        self._avg_win_turn = 100
        self._fastest_win = 100
        self._n_eliminations = 100
        self._avg_elim_turn = 100
        self._fastest_elim = 100

    def is_ranked(self):
        return self.played > 2

    def get_assists(self):
        return self._assists or ''

    def get_avg_win_turn(self):
        if self.wins == 0:
            return 'n/a'
        elif len(self.wins_by_turn) == 0:
            return '?'
        else:
            return self._avg_win_turn

    def get_fastest_win(self):
        if self.wins == 0:
            return 'n/a'
        elif len(self.wins_by_turn) == 0:
            return '?'
        else:
            return self._fastest_win

    def get_n_eliminations(self):
        if len(self.eliminations) == 0:
            return '?'
        else:
            return self._n_eliminations

    def get_avg_elim_turn(self):
        if len(self.eliminations) == 0:
            return '?'
        else:
            return self._avg_elim_turn

    def get_fastest_elim(self):
        if len(self.eliminations) == 0:
            return '?'
        else:
            return self._fastest_elim

    def update_game_results(self, game):
        if self.name == game.winner:
            if game.elims_by_deck:
                points = sum(map(
                    len,
                    game.elims_by_deck[self.name].values()
                ))
            else:
                points = len(game.losers)
            self.score += points
            self.wins += 1
        else:
            self.score -= 1
        self.played += 1
        self.games.append(game)

    def update_matchups(self, game):
        if self.name == game.winner:
            for loser_name in game.losers:
                if loser_name not in self.matchups:
                    self.matchups[loser_name] = 0
                self.matchups[loser_name] += 1
        else:
            if game.winner not in self.matchups:
                self.matchups[game.winner] = 0
            self.matchups[game.winner] -= 1

    def update_eliminations(self, game):
        if not game.elims_by_deck:
            return

        elims_by_turn = game.elims_by_deck[self.name]

        if self.name == game.winner:
            turn_won = max(elims_by_turn.keys())
            self.wins_by_turn[turn_won] += 1
        for turn, losers in elims_by_turn.items():
            n_elims = len(losers)
            self.eliminations[turn] += n_elims
            if self.name != game.winner:
                self._assists += n_elims
                self.score += n_elims

    def update_rank(self, metric, rank):
        r = self.ranks[metric]
        if r:
            delta = r - rank
            if delta == 0:
                self.rank_delta[metric] = ''
            else:
                self.rank_delta[metric] = ('+' if delta > 0 else '') + str(delta)
        else:
            self.rank_delta[metric] = 'New!'
        self.ranks[metric] = rank

    def calculate_metrics(self, decks):
        avg_n_players = statistics.mean(
            map(lambda game: len(game.decks), self.games)
        )
        self.winrate = self.wins / self.played
        self.expected_winrate = 1 / avg_n_players
        self.winrate_delta = self.winrate - self.expected_winrate

        if len(self.wins_by_turn):
            self._avg_win_turn = statistics.mean(self.wins_by_turn.elements())
            self._fastest_win = min(self.wins_by_turn.keys())

        if len(self.eliminations):
            self._n_eliminations = self.eliminations.total()
            self._avg_elim_turn = statistics.mean(self.eliminations.elements())
            self._fastest_elim = min(self.eliminations.keys())

        op_wins = sum(map(lambda op: decks[op].wins, self.matchups))
        op_played = sum(map(lambda op: decks[op].played, self.matchups))
        self.op_winrate = op_wins / op_played

        self.wrx1 = (self.winrate + 1) * (self.op_winrate + 1)
        # self.mu, self.sigma = mmr[self.name]


class Game:
    def __init__(self, date, decks_by_turn_order, winner, archideck_names, eliminations):
        self.date = date
        self.decks = decks_by_turn_order
        self.winner = winner
        self.losers = [d for d in self.decks if d != self.winner]
        self.elims_by_deck = None
        self._rankings = [
            [winner],
            self.losers
        ]

        if not eliminations:
            return

        self.elims_by_deck = {
            deck_name: {}
            for deck_name in archideck_names.values()
        }

        assists_by_deck = Counter({
            deck: 0
            for deck in decks_by_turn_order
            if deck != winner
        })

        for t, es in eliminations.items():
            turn = int(t)
            for e in es:
                eliminator = archideck_names[e['eliminator']]
                eliminated = list(map(lambda l: archideck_names[l], e['eliminated']))
                self.elims_by_deck[eliminator][turn] = eliminated
                if eliminator != winner:
                    assists_by_deck[eliminator] += len(eliminated)

        if not assists_by_deck.total():
            return

        self._rankings = [[winner]]

        decks_by_assists = [[], [], []]
        for deck, assists in assists_by_deck.items():
            decks_by_assists[assists].append(deck)
        for decks in reversed(decks_by_assists):
            if len(decks):
                self._rankings.append(sorted(decks, key=lambda d: decks_by_turn_order.index(d)))

    def get_rankings(self):
        decks = []
        rankings = []
        for r, ds in enumerate(self._rankings):
            for d in ds:
                decks.append(d)
                rankings.append(r)
        return decks, rankings


def parse_records(filepath):
    decks = {}
    games = {}
    with open(filepath, 'r') as f:
        records = json.load(f)

    valid_players_keys = {'player', 'deck', 'architect'}

    for record in records['games']:
        gs = record['games']
        date = record['date']
        games[date] = []

        for game in gs:
            players = game['players']
            winner = game['winner']
            simple_deck_names = list(map(lambda pxd: pxd['deck'], players))
            archideck_names = {}

            if winner not in simple_deck_names:
                raise Exception(f'winner {winner} not found in game dated {date}')

            for p in players:
                for key in p:
                    if key not in valid_players_keys:
                        raise Exception(f'invalid key {key} found in game dated {date}')
                player_name = p['player']
                simple_deck_name = p['deck']

                if 'architect' in p:
                    architect = p['architect']
                    if architect == player_name:
                        raise Exception(f'player {player_name} and architect match in game dated {date}')
                else:
                    architect = player_name

                archideck_name = simple_deck_name + ' - ' + architect
                archideck_names[simple_deck_name] = archideck_name
                if archideck_name not in decks:
                    decks[archideck_name] = Deck(simple_deck_name, architect)

            eliminations = {}

            if 'eliminations' in game:
                eliminations = game['eliminations']

                for t, es in eliminations.items():
                    for e in es:
                        eliminator = e['eliminator']
                        if eliminator not in simple_deck_names:
                            raise Exception(f'eliminator {eliminator} not found in game dated {date}')
                        eliminated = e['eliminated']
                        for loser in eliminated:
                            if loser not in simple_deck_names:
                                raise Exception(f'loser {loser} not found in game dated {date}')
                            if loser == winner:
                                raise Exception(f'winner {winner} eliminated in game dated {date}')

            game_object = Game(
                date,
                list(map(lambda sdn: archideck_names[sdn], simple_deck_names)),
                archideck_names[winner],
                archideck_names,
                eliminations
            )
            games[date].append(game_object)
    return decks, games


def update_record_results(decks, games):
    wins_by_turn_order = {
        3: Counter(),
        4: Counter()
    }

    games_as_list = [(d, gs) for d, gs in games.items()]

    for d, gs in games_as_list[:-1]:
        for game in gs:
            if parse_date(d) >= parse_date('2/2/2023'):
                wins_by_turn_order[len(game.decks)][game.decks.index(game.winner) + 1] += 1

            for deck_name in game.decks:
                deck = decks[deck_name]
                deck.update_game_results(game)
                deck.update_matchups(game)
                deck.update_eliminations(game)

            # deck_names, game_rankings = game.get_rankings()
            deck_names, _ = game.get_rankings()
            game_rankings = [0] + ([1] * len(game.losers))
            deck_ratings = [(decks[d].mmr,) for d in deck_names]
            new_ratings = rate(deck_ratings, ranks=game_rankings)
            for idx, (rating,) in enumerate(new_ratings):
                dd = deck_names[idx]
                ddd = decks[dd]
                ddd.mmr = rating

    last_game = games_as_list[-1]

    s = sorted(filter(lambda d: d.played, decks.values()), key=lambda d: d.name)

    for deck in s:
        deck.calculate_metrics(decks)

    _calculate_ranks(decks)

    for game in last_game[1]:
        wins_by_turn_order[len(game.decks)][game.decks.index(game.winner) + 1] += 1

        for deck_name in game.decks:
            deck = decks[deck_name]
            deck.update_game_results(game)
            deck.update_matchups(game)
            deck.update_eliminations(game)

        deck_names, game_rankings = game.get_rankings()
        deck_ratings = [(decks[d].mmr,) for d in deck_names]
        new_ratings = rate(deck_ratings, ranks=game_rankings)
        for idx, (rating,) in enumerate(new_ratings):
            decks[deck_names[idx]].mmr = rating

    for deck in decks.values():
        deck.calculate_metrics(decks)

    _calculate_ranks(decks)

    return wins_by_turn_order


def by_speed(d):
    if type(d.get_avg_win_turn()) is str or type(d.get_avg_elim_turn()) is str:
        return 100, 100, 100, 100, 100

    return d.get_avg_win_turn(), d.get_avg_elim_turn(), d.get_fastest_win(), d.get_fastest_elim(), -d.winrate


def _calculate_ranks(decks):
    ranked = list(filter(lambda d: d.is_ranked(), decks.values()))

    for metric, ranking in {
        'by_tbs': lambda d: (
                -d.winrate,
                -d.played,
                -d.op_winrate
        ),
        'by_delta': lambda d: (
                -d.winrate_delta,
                -d.played,
                -d.op_winrate
        ),
        'by_wrx1': lambda d: -d.wrx1,
        'by_mmr': lambda d: (
                -d.mmr.mu
        ),
        'by_speed': lambda d: (
                d._avg_win_turn,
                d._avg_elim_turn,
                d._fastest_win,
                d._fastest_elim,
                -d.winrate,
                -d._n_eliminations,
                -d.mmr.mu
        )
    }.items():
        for rank, deck in enumerate(sorted(ranked, key=ranking)):
            deck.update_rank(metric, rank+1)


def print_decks(decks, key):
    def print_deck(d):
        print(d.simple_name, d.architect, d.wins, d.played, d.winrate, d.expected_winrate, d.winrate_delta,
              d.get_fastest_win(), d.get_avg_win_turn(), d.get_n_eliminations(), d.get_fastest_elim(),
              d.get_avg_elim_turn(), d.get_assists(), d.score, d.op_winrate, d.wrx1,
              d.mmr.mu, d.mmr.sigma,
              d.ranks['by_mmr'], d.rank_delta['by_mmr'],
              d.ranks['by_delta'], d.rank_delta['by_delta'],
              d.ranks['by_tbs'], d.rank_delta['by_tbs'],
              d.ranks['by_wrx1'], d.rank_delta['by_wrx1'],
              d.ranks['by_speed'], d.rank_delta['by_speed'],
              sep='\t'
              )

    header = "name\tplayer\twins\tplayed\twin %\tex. win %\twin % delta\tfastest win\tavg win turn\teliminations" \
             "\tfastest elimination\tavg elim turn\tassists\tscore\top win %" \
             "\twrx1 ((win % + 1) x (op win % + 1))\tmmr\tsigma" \
             "\tBy mmr\tΔ" \
             "\tBy tiebreaks (win % delta, played, op win %)\tΔ" \
             "\tBy tiebreaks (win %, played, op win %)\tΔ" \
             "\tBy wrx1\tΔ" \
             "\tBy speed\tΔ"
    print(header)

    ranked, unranked = partition(lambda dck: dck.is_ranked(), sorted(decks.values(), key=key))

    for deck in ranked:
        print_deck(deck)

    print()

    for deck in unranked:
        print_deck(deck)

    print()


def foo(d):
    before = parse_date(d)
    with open('./data/edh.json', 'r',) as f:
        record = json.load(f)
    gs = record['games']

    decks = {}
    diffs = {}
    prev_date = ''
    wins_by_turn_order = {
        3: Counter(),
        4: Counter()
    }

    for g in gs:
        date = g['date']
        if parse_date(date) > before:
            return decks, diffs
        if date != prev_date:
            diffs[date] = {}
            prev_date = date
        games = g['games']
        for game in games:
            winner_name = game['winner']
            players = [p['deck'] for p in game['players']]
            losers = [d['deck'] for d in game['players'] if d['deck'] != winner_name]

            if parse_date(date) >= parse_date('2/2/2023'):
                wins_by_turn_order[len(players)][players.index(winner_name) + 1] += 1

            for deck_name in players:
                if deck_name not in decks:
                    decks[deck_name] = {
                        'name': deck_name,
                        'wins': 0,
                        'played': 0,
                        'score': 0,
                        'matchups': {},
                        'games': [],
                        'op_score': 0,
                        'wins_by_turn': Counter(),
                        'eliminations': Counter(),
                        'assists': 0
                    }

            decks[winner_name]['score'] += len(losers)

            for deck_name in players:
                deck = decks[deck_name]
                deck['played'] += 1
                # for op in players:
                #     if op != deck_name and op not in deck['matchups']:
                #         deck['matchups'][op] = 0

                if deck_name == winner_name:
                    deck['wins'] += 1

                    for loser_name in losers:
                        if loser_name not in deck['matchups']:
                            deck['matchups'][loser_name] = 0

                        deck['matchups'][loser_name] += 1
                        if deck_name not in diffs[date]:
                            diffs[date][deck_name] = {}

                        diffs[date][deck_name][loser_name] = deck['matchups'][loser_name]
                else:
                    if winner_name not in deck['matchups']:
                        deck['matchups'][winner_name] = 0
                    deck['matchups'][winner_name] -= 1

                    if deck_name not in diffs[date]:
                        diffs[date][deck_name] = {}

                    diffs[date][deck_name][winner_name] = deck['matchups'][winner_name]
                    deck['score'] -= 1

                if 'eliminations' in game:
                    eliminations = {
                        int(turn): turn_elims
                        for turn, turn_elims
                        in game['eliminations'].items()
                    }

                    if deck_name == winner_name:
                        turn_won = max(eliminations.keys())
                        deck['wins_by_turn'][turn_won] += 1

                    for t, e in eliminations.items():
                        for e_record in e:
                            if e_record['eliminator'] == deck_name:
                                deck['eliminations'][t] += len(e_record['eliminated'])
                                if deck_name != winner_name:
                                    deck['assists'] += 1

                game_object = {
                    'date': date,
                    'players': {
                        op: (1 if deck_name == winner_name else -1 if op == winner_name else 0) for op in
                        [p for p in players if p is not deck_name]
                    }
                }

                deck['games'].append(game_object)

    for deck in decks.values():
        deck['op_score'] = sum(map(lambda opr: decks[opr[0]]['score'] * opr[1], deck['matchups'].items()))

    return decks, diffs, wins_by_turn_order


def truskrillex(date=None):
    with open('./data/edh.json', 'r',) as f:
        record = json.load(f)
    gs = record['games']

    ratings_by_deck = {}
    ratings_by_deck_r = {}

    for g in gs:
        if date and parse_date(g['date']) >= parse_date(date):
            continue
        for game in g['games']:
            archideck_names = {
                p['deck']: p['deck'] + ' - ' + (p['player'] if 'architect' not in p else p['architect'])
                for p in game['players']
            }
            winner = archideck_names[game['winner']]
            decks = list(map(lambda p: archideck_names[p['deck']], game['players']))
            losers = list(filter(lambda dck: dck != winner, decks))

            for d in decks:
                if d not in ratings_by_deck:
                    ratings_by_deck[d] = Rating()

            rating_group = [(ratings_by_deck[winner],)] + [(ratings_by_deck[l],) for l in losers]
            rating_group_r = [(ratings_by_deck[winner],)] + list(reversed([(ratings_by_deck[l],) for l in losers]))
            ranks = [0] + ([1] * len(losers))
            new_ratings = rate(rating_group, ranks=ranks)
            new_ratings_r = rate(rating_group_r, ranks=ranks)
            test_ratings = rate(rating_group, ranks=list(range(len(decks))))
            for idx, (rating,) in enumerate(new_ratings):
                if idx == 0:
                    ratings_by_deck[winner] = rating
                else:
                    ratings_by_deck[losers[idx-1]] = rating

            for idx, (rating,) in enumerate(new_ratings_r):
                if idx == 0:
                    ratings_by_deck_r[winner] = rating
                else:
                    ratings_by_deck_r[losers[len(losers) - idx]] = rating

    # env = global_env()

    # for d, r in ratings_by_deck.items():
    #     print(f'{d}\t{env.expose(r)}')
    # print()

    # for deck, rating in sorted(ratings_by_deck.items(), key=lambda dxr: -dxr[1].mu):
    #     print(f'{deck}\t{rating.mu}\t{rating.sigma}')
    #
    # for deck, rating in sorted(ratings_by_deck_r.items(), key=lambda dxr: -dxr[1].mu):
    #     print(f'{deck}\t{rating.mu}\t{rating.sigma}')

    sort_by_mu = lambda dxr: -dxr[1].mu
    alphabetical = lambda dxr: dxr[0]

    # for (deck, rating), (_, rating_r) in zip(sorted(ratings_by_deck.items(), key=alphabetical),
    #                                          sorted(ratings_by_deck_r.items(), key=alphabetical)):
    #     print(f'{deck}\t{avg(rating.mu, rating_r.mu)}\t{avg(rating.sigma, rating_r.sigma)}')

    return {
        deck: (statistics.mean([rating.mu, rating_r.mu]), statistics.mean([rating.sigma, rating_r.sigma]))
        for (deck, rating), (_, rating_r)
        in zip(sorted(ratings_by_deck.items(), key=alphabetical),
               sorted(ratings_by_deck_r.items(), key=alphabetical))
    }


def print_wins(wins):
    for p in [3, 4]:
        print(f'{p}-player games')
        print('player #\twins\twin%')
        total = sum(wins[p].values())
        for i in range(1, p+1):
            ws = wins[p][i]
            print(f'{i}\t{ws}\t{ws / total}')


def print_diffs(decks, diffs):
    alphabetical = lambda kvp: kvp[0]
    by_score = lambda kvp: -decks[kvp[0]]['score']

    for date, ds in diffs.items():
        print(date)
        for deck, mus in sorted(ds.items(), key=by_score):
            print(f'{deck}\tvs.')
            for op, score in sorted(mus.items(), key=lambda oxs: -oxs[1]):
                print(f'{op}\t{score}')
            print()


def print_games_by_deck(decks):
    for deck in sorted(decks.values(), key=lambda d: d['name']):
        print(deck['name'], end='')
        for game in deck['games']:
            print('\t' + game['date'])
            for (op, result) in game['players'].items():
                print(f'{op}\t{result}')
        print()


def print_matchups(decks, key):
    for deck in sorted(decks.values(), key=key):
        print(f"{deck['name']}\tvs.")
        for op, score in sorted(deck['matchups'].items(), key=lambda kvp: kvp[0]):
            print(f'{op}\t{score}')
        print()


def print_graph(decks):
    edges = set()
    for deck in sorted(decks.values(), key=lambda d: -d['score']):
        for opponent, score in deck['matchups'].items():
            if score > 0:
                edges.add(f"  \"{deck['name']}\" -> \"{opponent}\"")
    print('digraph G {')
    for edge in edges:
        print(edge)
    print("}")


def calculate_scores(decks, trueskill_ratings):
    for deck in decks.values():

        avg_n_players = statistics.mean(
            map(lambda game: len(game['players']) + 1,
                deck['games']
                )
        )

        deck_name = deck['name']
        wins = deck['wins']
        played = deck['played']

        winrate = wins / played
        deck['winrate'] = winrate

        expected_winrate = 1 / avg_n_players
        deck['expected_winrate'] = expected_winrate

        winrate_delta = winrate - expected_winrate
        deck['winrate_delta'] = winrate_delta

        wins_by_turn = deck['wins_by_turn']
        if wins == 0:
            fastest_win = avg_turn_win = 'n/a'
        elif len(wins_by_turn) == 0:
            fastest_win = avg_turn_win = '?'
        else:
            fastest_win = min(wins_by_turn.keys())
            avg_turn_win = statistics.mean(wins_by_turn.elements())
        deck['fastest_win'] = fastest_win
        deck['avg_turn_win'] = avg_turn_win

        eliminations = deck['eliminations']
        if len(eliminations) == 0:
            elims = avg_elim_turn = fastest_elim = assists = '?'
        else:
            elims = eliminations.total()
            avg_elim_turn = statistics.mean(eliminations.elements())
            fastest_elim = min(eliminations.keys())
            assists = deck['assists'] if deck['assists'] else ''
        deck['elims'] = elims
        deck['fastest_elim'] = fastest_elim
        deck['avg_elim_turn'] = avg_elim_turn
        deck['assists'] = assists

        op_wins = deck['op_wins']
        op_lose = deck['op_lose']
        op_winrate = op_wins / (op_wins + op_lose)
        deck['op_winrate'] = op_winrate

        wrx = winrate * op_winrate
        wrx1 = (1+winrate) * (1+op_winrate)
        mu, sigma = trueskill_ratings[deck_name]

        deck['wrx1'] = wrx1
        deck['mu'] = mu
        deck['sigma'] = sigma


def rank_by(decks, rank_name, ranker):
    rank = 1
    for deck in sorted(decks, key=ranker):
        if deck['played'] > 2:
            deck[rank_name] = rank
            rank += 1
        else:
            deck[rank_name] = ''


def calculate_ranks(decks):
    def by_tbs(d):
        return -d['winrate'], -d['played'], -d['op_winrate']

    def by_delta(d):
        return -d['winrate_delta'], -d['played'], -d['op_winrate']

    ranked = decks.values()

    rank_by(ranked, 'by_tbs', by_tbs)
    rank_by(ranked, 'by_delta', by_delta)
    rank_by(ranked, 'by_wrx1', lambda d: -d['wrx1'])
    rank_by(ranked, 'by_mmr', lambda d: -d['mu'])


def print_scores(decks, key):
    print("name\twins\tplayed\twin %\tex. win %\twin % delta\tfastest win\tavg win turn\teliminations"
          "\tfastest elimination\tavg elim turn\tassists\tscore\top_score\top_wins\top_lose\top win %"
          "\twrx1 ((win % + 1) x (op win % + 1))\tmmr\tsigma\tBy tiebreaks (win %, played, op win %)"
          "\tBy tiebreaks (win % delta, played, op win %)\tBy wrx1\tBy mmr")
    for deck in sorted(decks.values(), key=key):
        print(f"{deck['name']}"
              f"\t{deck['wins']}"
              f"\t{deck['played']}"
              f"\t{deck['winrate']}"
              f"\t{deck['expected_winrate']}"
              f"\t{deck['winrate_delta']}"
              f"\t{deck['fastest_win']}"
              f"\t{deck['avg_turn_win']}"
              f"\t{deck['elims']}"
              f"\t{deck['fastest_elim']}"
              f"\t{deck['avg_elim_turn']}"
              f"\t{deck['assists']}"
              f"\t{deck['score']}"
              f"\t{deck['op_score']}"
              f"\t{deck['op_wins']}"
              f"\t{deck['op_lose']}"
              f"\t{deck['op_winrate']}"
              f"\t{deck['wrx1']}"
              f"\t{deck['mu']}"
              f"\t{deck['sigma']}"
              f"\t{deck['by_tbs']}"
              f"\t{deck['by_delta']}"
              f"\t{deck['by_wrx1']}"
              f"\t{deck['by_mmr']}"
            )


def opponents_wins_losses(decks):
    for deck in decks.values():
        opw = 0
        opl = 0
        for op in deck['matchups']:
            opw += decks[op]['wins']
            opl += decks[op]['played'] - decks[op]['wins']
        deck['op_wins'] = opw
        deck['op_lose'] = opl


def parse_date(d):
    m, d, y = [int(x) for x in d.split('/')]
    return datetime.date(y, m, d)


def partition(function, iterable):
    return filter(function, iterable), filter(lambda x: not function(x), iterable)


def main():
    by_score = lambda d: -d['score']
    alphabetical = lambda d: d.name

    record_filepath = './data/edh.json'

    date = "4/19/2999"
    decks, games = parse_records(record_filepath)
    wins = update_record_results(decks, games)
    print_decks(decks, key=alphabetical)
    print_wins(wins)

    # decks, diffs, wins = foo(date)
    # opponents_wins_losses(decks)
    # trueskill_ratings = trueskill()
    # calculate_scores(decks, trueskill_ratings)
    # calculate_ranks(decks)
    # print_scores(decks, key=alphabetical)
    # print_wins(wins)

    # print()
    # print_graph(decks)
    # print()
    # print_matchups(decks, key=lambda d: d['name'])
    # print()
    # print_games_by_deck(decks)
    # print()
    # print_diffs(decks, diffs)


if __name__ == '__main__':
    main()
