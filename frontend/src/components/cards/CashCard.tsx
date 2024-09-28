import { CashCardModel } from "@/types/cards";

type Props = {
  card: CashCardModel;
  count?: number;
};

export function CashCard({ card, count }: Props) {
  return (
    <div className="bg-white text-gray-900 rounded-lg p-3 border border-gray-200 w-[120px] h-[160px] flex flex-col justify-between relative">
      {/* Count badge in top right */}
      {count !== undefined && count > 1 && (
        <div className="absolute top-1 right-1 text-white text-[10px] font-bold rounded-full w-7 h-7 flex items-center justify-center border-2 border-white shadow-sm z-10 bg-blue-500">
          <span className="leading-none">{count}</span>
        </div>
      )}

      <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide py-1">Cash</div>

      <div className="text-center flex-1 flex items-center justify-center">
        <div className="text-2xl font-bold text-emerald-600">${card.value ?? "?"}</div>
      </div>

      <div></div>
    </div>
  );
}
