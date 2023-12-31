import datetime
import json
import math
import queue
import random
import statistics
from collections import Counter

import trueskill
from trueskill import Rating, rate
from trueskill.backends import cdf


class Deck:
    def __init__(self, name, architect):
        self.name = name + ' - ' + architect
        self.simple_name = name
        self.aliases = []
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
            'by_wr_delta': '',
            'by_wrx1': '',
            'by_mmr': '',
            'by_speed': '',
            'by_exposure': ''
        }
        self.rank_delta = {
            'by_tbs': '',
            'by_wr_delta': '',
            'by_wrx1': '',
            'by_mmr': '',
            'by_speed': '',
            'by_exposure': ''
        }
        self.winrate = None
        self.expected_winrate = None
        self.winrate_delta = None
        self.toaewr = None
        self.toaewr_delta = None
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

    def get_alias(self):
        if self.aliases:
            return self.aliases[0] + ' - ' + self.architect
        else:
            return self.name

    def get_simple_alias(self):
        if self.aliases:
            return self.aliases[0]
        else:
            return self.simple_name

    def get_assists(self):
        return self._assists or ''

    def get_avg_win_turn(self):
        if self.wins == 0:
            return ''
        elif len(self.wins_by_turn) == 0:
            return '?'
        else:
            return self._avg_win_turn

    def get_fastest_win(self):
        if self.wins == 0:
            return ''
        elif len(self.wins_by_turn) == 0:
            return '?'
        else:
            return self._fastest_win

    def get_n_eliminations(self):
        if len(self.eliminations) == 0:
            return ''
        else:
            return self._n_eliminations

    def get_avg_elim_turn(self):
        if len(self.eliminations) == 0:
            return ''
        else:
            return self._avg_elim_turn

    def get_fastest_elim(self):
        if len(self.eliminations) == 0:
            return ''
        else:
            return self._fastest_elim

    def update_game_results(self, game):
        if self.name in game.winners:
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
        if self.name in game.winners:
            for loser_name in game.losers:
                if loser_name not in self.matchups:
                    self.matchups[loser_name] = 0
                self.matchups[loser_name] += 1
        else:
            for winner in game.winners:
                if winner not in self.matchups:
                    self.matchups[winner] = 0
                self.matchups[winner] -= 1

    def update_eliminations(self, game):
        if not game.elims_by_deck:
            return

        elims_by_turn = game.elims_by_deck[self.name]

        if self.name in game.winners:
            turn_won = game.turns if not elims_by_turn else max(elims_by_turn.keys())
            self.wins_by_turn[turn_won] += 1
        for turn, losers in elims_by_turn.items():
            n_elims = len(losers)
            self.eliminations[turn] += n_elims
            if self.name not in game.winners:
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
            self.rank_delta[metric] = '!'
        self.ranks[metric] = rank

    def calculate_metrics(self, decks_by_names, wins_by_turn_order):
        avg_n_players = statistics.mean(
            map(lambda game: len(game.decks), self.games)
        )
        self.winrate = self.wins / self.played
        self.expected_winrate = 1 / avg_n_players
        self.winrate_delta = self.winrate - self.expected_winrate

        self._calc_turn_order_expected_winrate(wins_by_turn_order)

        if len(self.wins_by_turn):
            self._avg_win_turn = statistics.mean(self.wins_by_turn.elements())
            self._fastest_win = min(self.wins_by_turn.keys())

        if len(self.eliminations):
            self._n_eliminations = self.eliminations.total()
            self._avg_elim_turn = statistics.mean(self.eliminations.elements())
            self._fastest_elim = min(self.eliminations.keys())

        op_wins = sum(map(lambda op: decks_by_names[op].wins, self.matchups))
        op_played = sum(map(lambda op: decks_by_names[op].played, self.matchups))
        self.op_winrate = op_wins / op_played

        self.wrx1 = (self.winrate + 1) * (self.op_winrate + 1)

    def _calc_turn_order_expected_winrate(self, wins_by_turn_order):
        turn_position_counts = {
            3: Counter({
                i: 0 for i in range(1, 4)
            }),
            4: Counter({
                i: 0 for i in range(1, 5)
            }),
        }

        pod_size3 = 0
        pod_size4 = 0
        total_games = 0

        for game in self.games:
            if parse_date(game.date) >= parse_date('2/2/2023'):
                pod_size = len(game.decks)
                if pod_size == 3:
                    pod_size3 += 1
                elif pod_size == 4:
                    pod_size4 += 1
                else:
                    Exception(f'Game with pod size {pod_size} dated {game.date}')
                total_games += 1
                turn_position = game.decks.index(self.name) + 1
                turn_position_counts[pod_size][turn_position] += 1

        if not total_games:
            return 0

        turn_position_rates = {
            pod_size: {
                turn_position: 0 if not counts.total() else count / counts.total()
                for turn_position, count in counts.items()
            }
            for pod_size, counts in turn_position_counts.items()
        }

        pod_size3_ewr = sum(map(
            lambda pos: turn_position_rates[3][pos] * (wins_by_turn_order[3][pos]/wins_by_turn_order[3]['total']),
            range(1, 4))
        )
        pod_size4_ewr = sum(map(
            lambda pos: turn_position_rates[4][pos] * (wins_by_turn_order[4][pos]/wins_by_turn_order[4]['total']),
            range(1, 5))
        )

        self.toaewr = ((pod_size3 * pod_size3_ewr) + (pod_size4 * pod_size4_ewr))/total_games
        self.toaewr_delta = self.winrate - self.toaewr


class Game:
    def __init__(self, date, decks_by_turn_order, winners, archideck_names, eliminations):
        self.date = date
        self.decks = decks_by_turn_order
        self.winners = winners
        self.losers = [d for d in self.decks if d not in winners]
        self.elims_by_deck = None
        self.turns = None
        self._rankings = [
            winners,
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
            if deck not in winners
        })

        max_turn = 0

        for t, es in eliminations.items():
            turn = int(t)
            max_turn = max(max_turn, turn)
            for e in es:
                eliminator = archideck_names[e['eliminator']]
                eliminated = list(map(
                    lambda l: archideck_names[l],
                    ([] if 'eliminated' not in e else e['eliminated']) + ([] if 'scoops' not in e else e['scoops'])
                ))
                self.elims_by_deck[eliminator][turn] = eliminated
                if eliminator not in winners:
                    assists_by_deck[eliminator] += len(eliminated)

        self.turns = max_turn

        if not assists_by_deck.total():
            return

        self._rankings = [sorted(winners, key=decks_by_turn_order.index)]

        decks_by_assists = [[], [], []]
        for deck, assists in assists_by_deck.items():
            decks_by_assists[assists].append(deck)
        for decks in reversed(decks_by_assists):
            if len(decks):
                self._rankings.append(sorted(decks, key=decks_by_turn_order.index))

    def get_rankings(self):
        decks = []
        rankings = []
        for r, ds in enumerate(self._rankings):
            for d in ds:
                decks.append(d)
                rankings.append(r)
        return decks, rankings

    def update_results(self, decks_by_names, wins_by_turn_order):
        if parse_date(self.date) >= parse_date('2/2/2023'):
            pod_size = len(self.decks)
            wins_by_turn_order[pod_size]['total'] += 1
            for winner in self.winners:
                wins_by_turn_order[pod_size][self.decks.index(winner) + 1] += 1

        for deck_name in self.decks:
            deck = decks_by_names[deck_name]
            deck.update_game_results(self)
            deck.update_matchups(self)
            deck.update_eliminations(self)

        deck_names, game_rankings = self.get_rankings()
        deck_ratings = [(decks_by_names[dck].mmr,) for dck in deck_names]
        new_ratings = rate(deck_ratings, ranks=game_rankings)
        for idx, (rating,) in enumerate(new_ratings):
            decks_by_names[deck_names[idx]].mmr = rating


def parse_records(filepath, until_date=None):
    decks_by_names = {}
    games = []
    with open(filepath, 'r') as f:
        records = json.load(f)

    valid_players_keys = {'player', 'deck', 'architect', 'alias'}

    for record in records['games']:
        gs = record['games']
        date = record['date']
        if until_date and parse_date(until_date) <= parse_date(date):
            continue

        for game in gs:
            players = game['players']
            winners = game['winners']
            simple_deck_names = list(map(lambda pxd: pxd['deck'], players))
            archideck_names = {}

            for winner in winners:
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
                if archideck_name not in decks_by_names:
                    if 'alias' in p:
                        alias_name = p['alias'] + ' - ' + architect
                        if alias_name not in decks_by_names:
                            raise Exception(f'alias {alias_name} for {archideck_name} not found in game dated {date}')
                        aliased_deck = decks_by_names[alias_name]
                        aliased_deck.aliases.insert(0, simple_deck_name)
                        archideck_names[simple_deck_name] = alias_name
                        decks_by_names[archideck_name] = aliased_deck
                    else:
                        archideck_names[simple_deck_name] = archideck_name
                        decks_by_names[archideck_name] = Deck(simple_deck_name, architect)
                else:
                    archideck_names[simple_deck_name] = decks_by_names[archideck_name].name

            eliminations = {}

            if 'eliminations' in game:
                eliminations = game['eliminations']

                for t, es in eliminations.items():
                    for e in es:
                        eliminator = e['eliminator']
                        if eliminator not in simple_deck_names:
                            raise Exception(f'eliminator {eliminator} not found in game dated {date}')
                        if 'eliminated' in e:
                            eliminated = e['eliminated']
                            for loser in eliminated:
                                if loser not in simple_deck_names:
                                    raise Exception(f'loser {loser} not found in game dated {date}')
                                if loser in winners:
                                    raise Exception(f'winner {loser} eliminated in game dated {date}')
                        if 'scoops' in e:
                            scoops = e['scoops']
                            for loser in scoops:
                                if loser not in simple_deck_names:
                                    raise Exception(f'salty scoops {loser} not found in game dated {date}')
                                if loser in winners:
                                    raise Exception(f'winner {loser} eliminated in game dated {date}')
                        if 'eliminated' not in e and 'scoops' not in e:
                            raise Exception(f'losers not found in game dated {date}')

            game_object = Game(
                date,
                list(map(lambda sdn: archideck_names[sdn], simple_deck_names)),
                list(map(lambda wr: archideck_names[wr], winners)),
                archideck_names,
                eliminations
            )
            games.append(game_object)

    unique_decks = list({deck.get_alias(): deck for deck in decks_by_names.values()}.values())
    return games, unique_decks, decks_by_names


def update_record_results(games, unique_decks, decks_by_names):
    def update(gs, predicate=lambda _: True):
        for g in gs:
            g.update_results(decks_by_names, wins_by_turn_order)

        ds = [d for d in unique_decks if predicate(d)]
        for d in ds:
            d.calculate_metrics(decks_by_names, wins_by_turn_order)
        calculate_ranks(ds)

    wins_by_turn_order = {
        3: Counter(),
        4: Counter()
    }

    if len(games) == 1:
        update(games)
        return wins_by_turn_order

    last_date = games[-1].date
    past_games, new_games = partition(lambda g: g.date != last_date, games)

    update(past_games, predicate=lambda d: d.played)
    update(new_games)

    return wins_by_turn_order


def calculate_ranks(decks):
    env = trueskill.global_env()
    ranked = [d for d in decks if d.is_ranked()]

    for metric, ranking in {
        'by_tbs': lambda d: (
                -d.winrate,
                -d.played,
                -d.op_winrate
        ),
        'by_wr_delta': lambda d: (
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

    for rank, deck in enumerate(sorted(decks, key=lambda d: -env.expose(d.mmr))):
        deck.update_rank('by_exposure', rank+1)


def print_decks(decks, key):
    def print_deck(d):
        print(d.get_simple_alias(), d.architect, d.wins, d.played, d.winrate, d.expected_winrate, d.winrate_delta,
              '?' if d.toaewr is None else d.toaewr, '?' if d.toaewr_delta is None else d.toaewr_delta,
              d.get_fastest_win(), d.get_avg_win_turn(), d.get_n_eliminations(),
              d.get_fastest_elim(), d.get_avg_elim_turn(), d.get_assists(), d.score, d.op_winrate,
              d.wrx1, d.mmr.mu, d.mmr.sigma,
              d.ranks['by_exposure'], d.rank_delta['by_exposure'],
              d.ranks['by_mmr'], d.rank_delta['by_mmr'],
              d.ranks['by_wr_delta'], d.rank_delta['by_wr_delta'],
              d.ranks['by_wrx1'], d.rank_delta['by_wrx1'],
              d.ranks['by_speed'], d.rank_delta['by_speed'],
              sep='\t'
              )

    header = "name\tplayer\twins\tplayed\twin %\tex. win %\twin % delta" \
             "\tt.o. ex. win %\tt.o. ex. win % delta" \
             "\tfastest win\tavg win turn\teliminations" \
             "\tfastest elimination\tavg elim turn\tassists\tscore\top win %" \
             "\twrx1 ((win % + 1) x (op win % + 1))\tmmr\tsigma" \
             "\tBy exposure\tΔ" \
             "\tBy mmr\tΔ" \
             "\tBy win % delta (played, op win %)\tΔ" \
             "\tBy wrx1\tΔ" \
             "\tBy speed\tΔ"
    print(header)

    ranked, unranked = partition(lambda d: d.is_ranked(), sorted(decks, key=key))

    for deck in ranked:
        print_deck(deck)
    for deck in unranked:
        print_deck(deck)

    print()


def print_wins(wins):
    for p in [3, 4]:
        print(f'{p}-player games')
        print('player #\twins\twin%')
        total = wins[p]['total']
        for i in range(1, p+1):
            ws = wins[p][i]
            print(f'{i}\t{ws}\t{ws / total}')
        print(f'total\t{total}')
        print()


def parse_date(d):
    m, d, y = [int(x) for x in d.split('/')]
    return datetime.date(y, m, d)


def partition(function, iterable):
    return filter(function, iterable), filter(lambda x: not function(x), iterable)


def match_quality(decks, _print=False):
    env = trueskill.global_env()
    retired = ['Kess', 'Inalla', 'Teferi']
    filter_by_author = filter(lambda dck: dck.architect in ['June', 'Tony', 'Hooper', 'Pham', 'Rachael'], decks)
    filter_retired = filter(lambda dck: dck.simple_name not in retired, filter_by_author)
    filter_by_played = filter(lambda dck: dck.played > 2, filter_retired)
    decks = sorted(filter_by_played, key=lambda dck: dck.name)
    rev = [r for r in reversed(decks)]
    mqs = []
    if _print:
        print('\t', end='')
        for r in rev:
            print(f'{r.get_alias()}\t', end='')
        print()

    for d in decks:
        if _print:
            print(f'{d.get_alias()}\t', end='')
        for r in rev:
            if d.get_alias() == r.get_alias():
                break
            q = env.quality([(d.mmr,), (r.mmr,)])
            mqs.append((d.get_alias(), r.get_alias(), q))
            if _print:
                print(f'{q}\t', end='')
        if _print:
            print()
    return mqs


def mqs_to_grid(mqs):
    grid = {}
    for d, r, q in mqs:
        if d not in grid:
            grid[d] = []
        if r not in grid:
            grid[r] = []
        grid[d].append((q, r))
        grid[r].append((q, d))
    return grid


# matchmaking:
#   for each deck,
#       find best deck
#       if best deck not in group,
#           if deck not in group,
#               add deck to group
#           add best deck to deck's group
#       else
#           if deck not in group
#               add deck to best's group
#       else if more than 5 groups,
#           merge groups
def tiers_jank(grid):
    groups_by_deck = {}
    groups = []
    chaelela = 'Alela, Artful Provocateur - Rachael'
    chaeldrotha = 'Muldrotha - Rachael'

    for deck, qs in grid.items():
        by_q = sorted(qs, key=lambda qq: -qq[0])
        _, best = by_q[0]
        if best == chaeldrotha:
            _, best = by_q[1]
            # if best == chaeldrotha:
            #     _, best = by_q[2]
        if best not in groups_by_deck:
            if deck not in groups_by_deck:
                group = len(groups)
                groups.append([deck])
                groups_by_deck[deck] = group
            group = groups_by_deck[deck]
            groups_by_deck[best] = group
            groups[group].append(best)
        elif deck not in groups_by_deck:
            best_group = groups_by_deck[best]
            groups_by_deck[deck] = best_group
            groups[best_group].append(deck)
        elif len(groups) > 5:
            group = groups_by_deck[deck]
            best_group = groups_by_deck[best]
            best_group_list = groups.pop(best_group)
            groups[group] += best_group_list
            for d in best_group_list:
                groups_by_deck[d] = group
    sorted_groups = [sorted(g) for g in groups]
    print_tier_groups(sorted_groups)


def tiers(mqs):
    grid = mqs_to_grid(mqs)
    # for deck, qs in grid.items():
    #     by_q = sorted(qs, key=lambda qq: -qq[0])
    #     _, best = by_q[0]
    #     print(f'{deck}\t{best}')
    faves = {
        deck: sorted(qq, key=lambda qqq: -qqq[0])[0][1]
        for deck, qq in grid.items()
    }
    popularity = Counter(faves.values())
    by_faves = {}
    for d, f in faves.items():
        if f not in by_faves:
            by_faves[f] = []
        by_faves[f].append(d)

    groups = [
        [f] + sorted(ds)
        for f, ds in by_faves.items()
    ]
    return groups


def print_tier_groups_graph(groups):
    print('digraph G {')
    for group in groups:
        for deck in group[1:]:
            print(f'\t\"{deck}\" -> \"{group[0]}\"')
    print('}')


def print_tier_groups(groups):
    for group in groups:
        print(group)
    for i in range(max(map(len, groups))):
        for group in groups:
            deck = ''
            arch = ''
            if i < len(group):
                archideck = group[i]
                deck, arch = archideck.split(' - ')
            print(deck, end='\t')
            print(arch, end='\t\t')
        print()


def pods(decks):
    env = trueskill.global_env()
    decks = [d for d in decks if d.architect in {'Hooper', 'June', 'Pham', 'Rachael', 'Tony'}]

    unique_players = ['Hooper', 'June', 'Pham', 'Rachael', 'Tony']
    upqs = []
    for idx in range(len(unique_players)):
        pod = unique_players[:idx] + unique_players[idx+1:]
        for di in filter(lambda d: d.architect == pod[0], decks):
            for dj in filter(lambda d: d.architect == pod[1], decks):
                for dk in filter(lambda d: d.architect == pod[2], decks):
                    for dl in filter(lambda d: d.architect == pod[3], decks):
                        ds = [di, dj, dk, dl]
                        q = env.quality(list(map(lambda d: (d.mmr,), ds)))
                        upqs.append((list(map(lambda d: d.get_alias(), ds)), q))

    unique_pods_by_best = sorted(upqs, key=lambda dsq: dsq[1])

    for ds, q in unique_pods_by_best[:10]:
        print(f'{q}\t{ds}')

    for ds, q in unique_pods_by_best[-10:]:
        print(f'{q}\t{ds}')

    pqs = []
    for i in range(len(decks)):
        for j in range(i+1, len(decks)):
            for k in range(j+1, len(decks)):
                for l in range(k+1, len(decks)):
                    ds = [decks[i], decks[j], decks[k], decks[l]]
                    q = env.quality(list(map(lambda d: (d.mmr,), ds)))
                    pqs.append((list(map(lambda d: d.get_alias(), ds)), q))

    pods_by_best = sorted(pqs, key=lambda dsq: dsq[1])

    for ds, q in pods_by_best[:10]:
        print(f'{q}\t{ds}')

    for ds, q in pods_by_best[-10:]:
        print(f'{q}\t{ds}')


def exposure_tiers(decks):
    env = trueskill.global_env()
    retired = ['Inalla', 'Kess', 'Teferi']
    filter_by_author = filter(lambda dck: dck.architect in ['June', 'Tony', 'Hooper', 'Pham', 'Rachael'], decks)
    filter_by_played = filter(lambda dck: dck.played > 2, filter_by_author)
    filter_by_retired = filter(lambda dck: dck.simple_name not in retired, filter_by_played)
    by_exposure = sorted(filter_by_retired, key=lambda d: d.ranks['by_exposure'])
    worst_mqs = queue.PriorityQueue()
    exposure_tiers_by_deck = {}
    for idx in range(1, len(by_exposure)):
        d1 = decks[idx-1]
        d2 = decks[idx]
        mq = env.quality([(d1.mmr,), (d2.mmr,)])
        worst_mqs.put((mq, idx))
    tier_breakpoints = []
    for i in range(4):
        tier_breakpoints.append(worst_mqs.get()[1])
    tier_breakpoints = sorted(tier_breakpoints, reverse=True)
    next_breakpoint = tier_breakpoints.pop()
    tier = 1
    for idx, deck in enumerate(by_exposure):
        if idx == next_breakpoint:
            if len(tier_breakpoints):
                next_breakpoint = tier_breakpoints.pop()
            tier += 1
        exposure_tiers_by_deck[deck.get_alias()] = tier
    for deck, tier in exposure_tiers_by_deck.items():
        print(f'{deck}\t{tier}')


def win_probability(a, b):
    env = trueskill.global_env()
    deltaMu = sum([x.mu for x in a]) - sum([x.mu for x in b])
    sumSigma = sum([x.sigma ** 2 for x in a]) + sum([x.sigma ** 2 for x in b])
    playerCount = len(a) + len(b)
    denominator = math.sqrt(playerCount * (trueskill.BETA * trueskill.BETA) + sumSigma)
    return env.cdf(deltaMu / denominator)


def build_pods(decks):
    decks = sorted(decks, key=lambda d: -d.played)
    max_games_played = decks[0].played
    for deck in decks[1:]:
        while deck.played < max_games_played:
            ops = [d for d in decks if deck.name != d.name and d.played < max_games_played]
            random.shuffle(ops)
            pod_size = min(random.randint(3, 4), len(ops))
            if pod_size < 3:
                break
            pod = [deck]
            while len(pod) < pod_size:
                pod.append(ops.pop())

            # pod_names = [d.name for d in pod]

            if Counter([dd.simple_name for dd in pod]).most_common(1)[0][1] > 1:
                continue

            yield pod


def sim(unique_decks, decks_by_names):
    date = '12/2/2023'
    for pod in build_pods(unique_decks):
        random.shuffle(pod)
        names_in_pod_order = [dck.name for dck in pod]
        print(names_in_pod_order)

        winrates = {}
        remaining = set(pod)
        eliminations = {}
        for d in pod:
            wrs = []
            for o in pod:
                if d == o:
                    continue
                wrs.append((o, win_probability((d.mmr,), (o.mmr,))))
            winrates[d] = sorted(wrs, key=lambda owr: -owr[1])
        turn = min(map(lambda dck: dck.get_fastest_win() if type(dck.get_fastest_win()) is int else 100, pod))
        while len(remaining) > 1:
            eliminations[turn] = []
            elims_this_turn = {}
            for d in pod:
                if d not in remaining:
                    continue
                for o, wr in winrates[d]:
                    if o not in remaining:
                        continue
                    result = random.random()
                    if result < wr:
                        if d not in elims_this_turn:
                            elims_this_turn[d.simple_name] = []
                        elims_this_turn[d.simple_name].append(o.simple_name)
                        remaining.remove(o)
            for eliminator, eliminated in elims_this_turn.items():
                eliminations[turn].append({
                    'eliminator': eliminator,
                    'eliminated': eliminated
                })
            turn += 1
        game = Game(
            date,
            [dck.name for dck in pod],
            [dck.name for dck in remaining],
            {dck.simple_name: dck.name for dck in pod},
            eliminations
        )
        update_record_results([game], unique_decks, decks_by_names)
        print(list(remaining)[0].name)
        print(eliminations)
        print()


def main():
    trueskill.setup(draw_probability=0.001)

    by_score = lambda d: -d['score']
    alphabetical = lambda d: d.get_alias()

    record_filepath = './data/edh.json'

    date = "12/31/2999"
    games, unique_decks, decks_by_names = parse_records(record_filepath, until_date=date)
    wins = update_record_results(games, unique_decks, decks_by_names)
    print_decks(unique_decks, key=alphabetical)
    print_wins(wins)

    # pods(unique_decks)

    # random.seed(8675309)
    # sim(unique_decks, decks_by_names)

    # t = tiers(match_quality(unique_decks))
    # print_tier_groups_graph(t)
    # print_tier_groups(t)

    # exposure_tiers(unique_decks)
    # for d, r, q in match_quality(unique_decks):
    #     print(f'{d}\t{r}\t{q}')

    # print_match_quality(unique_decks)


if __name__ == '__main__':
    main()
