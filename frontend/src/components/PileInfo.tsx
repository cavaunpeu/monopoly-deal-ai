export type PileSizes = {
    deck: number;
    discard: number;
};

type Props = {
    piles: PileSizes;
};

export function PileInfo({ piles }: Props) {
    return (
        <div className="border border-gray-200 rounded-lg p-3 bg-white">
            <div className="text-sm text-gray-700 flex items-center gap-4">
                <span>Deck: {piles.deck} cards</span>
                <span>Discard: {piles.discard} cards</span>
            </div>
        </div>
    );
}